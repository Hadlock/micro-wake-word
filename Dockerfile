# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MICROWAKEWORD_WORKDIR=/workspace \
    MICROWAKEWORD_VOICE_DIR=/opt/piper-voices \
    MICROWAKEWORD_VOICE_MODEL_NAME=en_US-lessac-medium.onnx \
    MICROWAKEWORD_VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true" \
    MICROWAKEWORD_VOICE_CONFIG_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true"

ENV MICROWAKEWORD_VOICE_MODEL="${MICROWAKEWORD_VOICE_DIR}/${MICROWAKEWORD_VOICE_MODEL_NAME}" \
    MICROWAKEWORD_VOICE_CONFIG="${MICROWAKEWORD_VOICE_MODEL}.json"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        unzip \
        libsndfile1 \
        ffmpeg \
    espeak-ng \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

# Install dependencies in order:
# 1. PyTorch CPU (brings compatible numpy)
# 2. TensorFlow (will upgrade to numpy 2.x as needed by tf>=2.16)
# 3. onnxruntime 1.19+ (supports numpy 2.x)
# 4. Piper dependencies + datasets[audio] extra for complete audio support
# 5. microwakeword itself (last, so tensorflow requirement is already satisfied)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1+cpu \
        torchaudio==2.4.1+cpu && \
    pip install --no-cache-dir \
        'onnxruntime>=1.19.0' \
        'piper-tts==1.2.0' \
        'piper-phonemize-cross==1.2.1' \
        'git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f' \
        'datasets[audio]' && \
    pip install --no-cache-dir -e .

RUN mkdir -p ${MICROWAKEWORD_WORKDIR} /app/serve ${MICROWAKEWORD_VOICE_DIR} \
    && wget -O ${MICROWAKEWORD_VOICE_MODEL} ${MICROWAKEWORD_VOICE_URL} \
    && wget -O ${MICROWAKEWORD_VOICE_CONFIG} ${MICROWAKEWORD_VOICE_CONFIG_URL}

EXPOSE 8080

ENTRYPOINT ["python", "-m", "microwakeword.docker_entrypoint"]
