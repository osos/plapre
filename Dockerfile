ARG CUDA_IMAGE=nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    libsndfile1 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel

COPY pyproject.toml README.md /app/
COPY plapre /app/plapre

RUN pip install ".[serve]"

ENV PLAPRE_CHECKPOINT=syvai/plapre-nano \
    PLAPRE_GPU_MEM=0.5 \
    PLAPRE_MAX_MODEL_LEN=512 \
    PLAPRE_ASYNC=1

EXPOSE 8000

CMD ["plapre-serve", "--host", "0.0.0.0", "--port", "8000"]
