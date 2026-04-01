"""
FastAPI server for Plapre Danish TTS with buffered/streamed PCM/WAV output.

Start with:
    plapre-serve --port 8000

Or:
    uvicorn plapre.server:app
"""

import asyncio
import io
import logging
import os
import struct
import wave
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import AliasChoices, BaseModel, Field

from plapre.inference import SAMPLE_RATE, Plapre

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_tts: Plapre | None = None
_vocoder_sem: asyncio.Semaphore | None = None
_async_mode: bool = False
_response_mode_default: Literal["buffered", "stream"] = "buffered"

SUPPORTED_MODES_BY_FORMAT: dict[str, set[str]] = {
    "pcm": {"buffered", "stream"},
    "wav": {"buffered", "stream"},
}
DEFAULT_RESPONSE_MODE: Literal["buffered", "stream"] = "buffered"
WAV_STREAM_PLACEHOLDER_SIZE = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tts, _vocoder_sem, _async_mode, _response_mode_default

    checkpoint = os.environ.get("PLAPRE_CHECKPOINT", "syvai/plapre-nano")
    quant = os.environ.get("PLAPRE_QUANT", "q8_0")
    dtype = os.environ.get("PLAPRE_DTYPE", "auto")
    gpu_mem = float(os.environ.get("PLAPRE_GPU_MEM", "0.5"))
    max_len = int(os.environ.get("PLAPRE_MAX_MODEL_LEN", "512"))
    _async_mode = os.environ.get("PLAPRE_ASYNC", "1") == "1"
    mode_raw = os.environ.get("PLAPRE_RESPONSE_MODE_DEFAULT", DEFAULT_RESPONSE_MODE)
    mode = mode_raw.strip().lower()
    if mode not in {"buffered", "stream"}:
        log.warning(
            "Invalid PLAPRE_RESPONSE_MODE_DEFAULT=%r, using %r",
            mode_raw,
            DEFAULT_RESPONSE_MODE,
        )
        mode = DEFAULT_RESPONSE_MODE
    _response_mode_default = mode
    mode_str = "async" if _async_mode else "sync"
    log.info(
        "Loading model %s (quant=%s, dtype=%s, gpu_mem=%.2f, max_len=%d, mode=%s, response_mode_default=%s) …",
        checkpoint,
        quant,
        dtype,
        gpu_mem,
        max_len,
        mode_str,
        _response_mode_default,
    )
    _tts = Plapre(
        checkpoint=checkpoint,
        quant=quant,
        dtype=dtype,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_len,
        use_async=_async_mode,
    )
    # Serialize vocoder calls — Vocos cuFFT needs exclusive GPU access to avoid OOM
    _vocoder_sem = asyncio.Semaphore(1)
    log.info("Model ready (%s mode).", mode_str)
    yield
    _tts = None


app = FastAPI(title="Plapre TTS", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    text: str = Field(validation_alias=AliasChoices("text", "input"))
    speaker: str | None = Field(
        default=None,
        validation_alias=AliasChoices("speaker", "voice"),
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "pcm"
    stream_format: Literal["sse", "audio"] = "audio"
    response_mode: Literal["buffered", "stream"] | None = None
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] audio to 16-bit signed LE PCM bytes."""
    arr = np.asarray(audio, dtype=np.float32)

    # Defensively normalize shape to mono to avoid malformed byte layout
    # if an upstream vocoder ever returns channel dimensions.
    if arr.ndim == 2:
        # Heuristic: small leading dimension is likely channel-first.
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=1)
    elif arr.ndim > 2:
        arr = np.squeeze(arr)
        if arr.ndim != 1:
            arr = arr.reshape(-1)

    clipped = np.clip(arr, -1.0, 1.0)
    # Use explicit little-endian int16 independent of host endianness.
    pcm = (clipped * 32767.0).astype("<i2")
    return pcm.tobytes()


def _audio_headers() -> dict[str, str]:
    return {
        "X-Sample-Rate": str(SAMPLE_RATE),
        "X-Channels": "1",
        "X-Bit-Depth": "16",
    }


def _pcm16_to_wav(pcm16_bytes: bytes) -> bytes:
    """Wrap mono 16-bit PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm16_bytes)
    return buf.getvalue()


def _wav_stream_header() -> bytes:
    """Build a WAV header with unknown total data length for live streaming."""
    channels = 1
    bits_per_sample = 16
    bytes_per_sample = bits_per_sample // 8
    block_align = channels * bytes_per_sample
    byte_rate = SAMPLE_RATE * block_align
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", WAV_STREAM_PLACEHOLDER_SIZE),
            b"WAVE",
            b"fmt ",
            struct.pack("<I", 16),
            struct.pack(
                "<HHIIHH",
                1,  # PCM
                channels,
                SAMPLE_RATE,
                byte_rate,
                block_align,
                bits_per_sample,
            ),
            b"data",
            struct.pack("<I", WAV_STREAM_PLACEHOLDER_SIZE),
        ]
    )


async def _collect_audio(chunks: AsyncGenerator[bytes, None]) -> bytes:
    parts: list[bytes] = []
    async for chunk in chunks:
        parts.append(chunk)
    return b"".join(parts)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if _tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if req.stream_format != "audio":
        raise HTTPException(
            status_code=400,
            detail='Unsupported stream_format: only "audio" is currently supported',
        )
    if req.response_format not in SUPPORTED_MODES_BY_FORMAT:
        raise HTTPException(
            status_code=400,
            detail='Unsupported response_format: only "pcm" and "wav" are currently supported',
        )
    response_mode = (req.response_mode or _response_mode_default).strip().lower()
    supported_modes = SUPPORTED_MODES_BY_FORMAT[req.response_format]
    if response_mode not in supported_modes:
        raise HTTPException(
            status_code=400,
            detail=(
                f'Unsupported response_mode "{response_mode}" for response_format '
                f'"{req.response_format}". Supported modes: {sorted(supported_modes)}'
            ),
        )

    try:
        spk = _tts._resolve_speaker(req.speaker, None, None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    gen_kwargs = dict(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_tokens=req.max_tokens,
    )

    sentences = _tts._split_sentences(req.text)
    if not sentences:
        raise HTTPException(status_code=400, detail="No text provided")

    silence_samples = int(0.1 * SAMPLE_RATE)
    silence_bytes = struct.pack(f"<{silence_samples}h", *([0] * silence_samples))

    async def generate_pcm() -> AsyncGenerator[bytes, None]:
        if _async_mode:
            # Async: all sentences submitted concurrently, vLLM batches internally
            log.info("Generating %d sentence(s) via AsyncLLM", len(sentences))
            all_tokens = await _tts.generate_tokens_async(
                sentences, spk, **gen_kwargs
            )
        else:
            # Sync: batch via vLLM sync engine on thread pool
            log.info("Generating %d sentence(s) via sync LLM", len(sentences))
            all_tokens = await asyncio.to_thread(
                _tts._generate_tokens, sentences, spk, **gen_kwargs
            )

        # Vocode sequentially via semaphore — cuFFT needs exclusive GPU access
        for i, tokens in enumerate(all_tokens):
            async with _vocoder_sem:
                try:
                    audio = await asyncio.to_thread(
                        _tts._tokens_to_audio, tokens, spk
                    )
                except (torch.OutOfMemoryError, RuntimeError) as e:
                    log.warning("Vocoder failed for sentence %d: %s", i, e)
                    audio = None
            if audio is not None:
                yield _float32_to_pcm16(audio)
                if i < len(sentences) - 1:
                    yield silence_bytes

    if req.response_format == "pcm":
        if response_mode == "stream":
            return StreamingResponse(
                generate_pcm(),
                media_type="audio/pcm",
                headers=_audio_headers(),
            )
        pcm_bytes = await _collect_audio(generate_pcm())
        return Response(
            content=pcm_bytes,
            media_type="audio/pcm",
            headers=_audio_headers(),
        )

    # req.response_format == "wav"
    if response_mode == "buffered":
        pcm_bytes = await _collect_audio(generate_pcm())
        wav_bytes = _pcm16_to_wav(pcm_bytes)
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers=_audio_headers(),
        )

    async def generate_wav_stream() -> AsyncGenerator[bytes, None]:
        yield _wav_stream_header()
        async for chunk in generate_pcm():
            yield chunk

    return StreamingResponse(
        generate_wav_stream(),
        media_type="audio/wav",
        headers=_audio_headers(),
    )


@app.get("/v1/speakers")
async def speakers():
    if _tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"speakers": list(_tts.speakers.keys())}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Plapre TTS server")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="HuggingFace checkpoint (overrides PLAPRE_CHECKPOINT when provided)",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help='vLLM dtype (e.g. "auto", "float16", "bfloat16"; overrides PLAPRE_DTYPE when provided)',
    )
    parser.add_argument(
        "--gpu-mem",
        type=float,
        default=None,
        help="GPU memory utilization (overrides PLAPRE_GPU_MEM when provided)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Max model length (overrides PLAPRE_MAX_MODEL_LEN when provided)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        default=None,
        help="Use sync vLLM engine (sets PLAPRE_ASYNC=0 when provided)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    if args.checkpoint is not None:
        os.environ["PLAPRE_CHECKPOINT"] = args.checkpoint
    if args.dtype is not None:
        os.environ["PLAPRE_DTYPE"] = args.dtype
    if args.gpu_mem is not None:
        os.environ["PLAPRE_GPU_MEM"] = str(args.gpu_mem)
    if args.max_model_len is not None:
        os.environ["PLAPRE_MAX_MODEL_LEN"] = str(args.max_model_len)
    if args.sync is True:
        os.environ["PLAPRE_ASYNC"] = "0"
    uvicorn.run(
        "plapre.server:app",
        host=args.host,
        port=args.port,
        http="httptools",
    )


if __name__ == "__main__":
    main()
