# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MICROWAKEWORD_WORKDIR=/workspace \
    PIPER_HOME=/opt/piper-sample-generator

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        unzip \
        libsndfile1 \
        ffmpeg \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1+cpu torchaudio==2.4.1+cpu \
    && pip install --no-cache-dir piper-phonemize-cross==1.2.1 \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir 'git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f'

RUN git clone https://github.com/rhasspy/piper-sample-generator ${PIPER_HOME} \
    && mkdir -p ${PIPER_HOME}/models \
    && wget -O ${PIPER_HOME}/models/en_US-libritts_r-medium.pt \
        https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt

RUN mkdir -p ${MICROWAKEWORD_WORKDIR} /app/serve

EXPOSE 8080

ENTRYPOINT ["python", "-m", "microwakeword.docker_entrypoint"]
