"""Container entrypoint for training microWakeWord models on CPU."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import timedelta
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import zipfile

import yaml

from mmap_ninja.ragged import RaggedMmap  # type: ignore[import-not-found]

from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration

NEGATIVE_DATASET_ROOT = (
    "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
)
NEGATIVE_DATASETS = {
    "dinner_party.zip": "dinner_party",
    "dinner_party_eval.zip": "dinner_party_eval",
    "no_speech.zip": "no_speech",
    "speech.zip": "speech",
}
DEFAULT_SAMPLE_COUNT = int(os.getenv("MICROWAKEWORD_SAMPLE_COUNT", "400"))
DEFAULT_BATCH_SIZE = int(os.getenv("MICROWAKEWORD_SAMPLE_BATCH", "50"))
DEFAULT_TRAINING_STEPS = int(os.getenv("MICROWAKEWORD_TRAINING_STEPS", "10000"))
DEFAULT_WORKDIR = Path(os.getenv("MICROWAKEWORD_WORKDIR", "/workspace"))
PIPER_HOME = Path(os.getenv("PIPER_HOME", "/opt/piper-sample-generator"))


class UTCFormatter(logging.Formatter):
    """Formatter that forces UTC timestamps."""

    def formatTime(self, record, datefmt=None):  # noqa: D401 (inherits docstring)
        dt = time.gmtime(record.created)
        if datefmt:
            return time.strftime(datefmt, dt) + "Z"
        return time.strftime("%Y-%m-%dT%H:%M:%S", dt) + "Z"


def configure_logging() -> None:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(UTCFormatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


def slugify_phrase(wakeword: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", wakeword.lower()).strip("_")
    return slug or "wakeword"


def run_command(command: list[str], *, cwd: Path | None = None) -> None:
    logging.debug("Running command: %s", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def ensure_wakeword_samples(
    *, wakeword: str, samples_dir: Path, max_samples: int, batch_size: int
) -> None:
    samples_dir.mkdir(parents=True, exist_ok=True)
    if list(samples_dir.glob("*.wav")):
        logging.info("wake word samples already exist; skipping synthesis")
        return

    if not PIPER_HOME.exists():
        raise FileNotFoundError(
            f"Missing piper-sample-generator at {PIPER_HOME}. Container build may be incomplete."
        )

    logging.info(
        "generating %d synthetic samples for '%s' using piper", max_samples, wakeword
    )
    cmd = [
        sys.executable,
        "generate_samples.py",
        wakeword,
        "--max-samples",
        str(max_samples),
        "--batch-size",
        str(batch_size),
        "--output-dir",
        str(samples_dir),
    ]
    run_command(cmd, cwd=PIPER_HOME)


def download_file(url: str, destination: Path) -> None:
    import urllib.request

    destination.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(urllib.request.urlopen(url)) as response, destination.open(
        "wb"
    ) as output:
        shutil.copyfileobj(response, output)


def ensure_negative_datasets(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for archive, folder in NEGATIVE_DATASETS.items():
        target_dir = base_dir / folder
        manifest = target_dir / "wakeword_mmap" / "manifest.json"
        if manifest.exists():
            logging.info("negative dataset '%s' already present", folder)
            continue

        url = NEGATIVE_DATASET_ROOT + archive
        archive_path = base_dir / archive
        if not archive_path.exists():
            logging.info("downloading %s", archive)
            download_file(url, archive_path)
        logging.info("extracting %s", archive)
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(base_dir)


def generate_positive_feature_sets(samples_dir: Path, features_dir: Path) -> None:
    features_dir.mkdir(parents=True, exist_ok=True)
    clips = Clips(
        input_directory=str(samples_dir),
        file_pattern="*.wav",
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )

    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.05,
            "TanhDistortion": 0.05,
            "PitchShift": 0.05,
            "BandStopFilter": 0.05,
            "AddColorNoise": 0.05,
            "AddBackgroundNoise": 0.0,
            "Gain": 1.0,
            "RIR": 0.0,
        },
        impulse_paths=[],
        background_paths=[],
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )

    for split in ("training", "validation", "testing"):
        split_dir = features_dir / split
        mmap_dir = split_dir / "wakeword_mmap"
        manifest = mmap_dir / "manifest.json"
        if manifest.exists():
            logging.info("positive features for %s already exist", split)
            continue

        split_dir.mkdir(parents=True, exist_ok=True)

        if split == "training":
            split_name = "train"
            repetition = 2
            spectrograms = SpectrogramGeneration(
                clips=clips, augmenter=augmenter, slide_frames=10, step_ms=10
            )
        elif split == "validation":
            split_name = "validation"
            repetition = 1
            spectrograms = SpectrogramGeneration(
                clips=clips, augmenter=augmenter, slide_frames=10, step_ms=10
            )
        else:
            split_name = "test"
            repetition = 1
            spectrograms = SpectrogramGeneration(
                clips=clips, augmenter=augmenter, slide_frames=1, step_ms=10
            )

        logging.info("creating positive feature set for %s", split)
        RaggedMmap.from_generator(
            out_dir=str(mmap_dir),
            sample_generator=spectrograms.spectrogram_generator(
                split=split_name, repeat=repetition
            ),
            batch_size=100,
            verbose=True,
        )


def write_training_config(
    *, session_dir: Path, slug: str, training_steps: int
) -> tuple[Path, Path]:
    train_dir = Path("trained_models") / slug

    config = {
        "window_step_ms": 10,
        "train_dir": str(train_dir),
        "features": [
            {
                "features_dir": "generated_augmented_features",
                "sampling_weight": 2.0,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            },
            {
                "features_dir": "negative_datasets/speech",
                "sampling_weight": 10.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": "negative_datasets/dinner_party",
                "sampling_weight": 10.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": "negative_datasets/no_speech",
                "sampling_weight": 5.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": "negative_datasets/dinner_party_eval",
                "sampling_weight": 0.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "split",
                "type": "mmap",
            },
        ],
        "training_steps": [training_steps],
        "positive_class_weight": [1],
        "negative_class_weight": [20],
        "learning_rates": [0.001],
        "batch_size": 128,
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": 500,
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }

    config_path = session_dir / "training_parameters.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return config_path, session_dir / train_dir


def run_training_process(config_path: Path, *, workdir: Path) -> None:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    command = [
        sys.executable,
        "-m",
        "microwakeword.model_train_eval",
        f"--training_config={config_path.name}",
        "--train",
        "1",
        "--restore_checkpoint",
        "1",
        "--test_tf_nonstreaming",
        "0",
        "--test_tflite_nonstreaming",
        "0",
        "--test_tflite_nonstreaming_quantized",
        "0",
        "--test_tflite_streaming",
        "0",
        "--test_tflite_streaming_quantized",
        "1",
        "--use_weights",
        "best_weights",
        "mixednet",
        "--pointwise_filters",
        "64,64,64,64",
        "--repeat_in_block",
        "1,1,1,1",
        "--mixconv_kernel_sizes",
        "[5],[7,11],[9,15],[23]",
        "--residual_connection",
        "0,0,0,0",
        "--first_conv_filters",
        "32",
        "--first_conv_kernel_size",
        "5",
        "--stride",
        "3",
    ]

    logging.info("launching training process")
    subprocess.run(command, cwd=workdir, check=True, env=env)


def locate_tflite_model(train_dir: Path) -> Path:
    candidate = (
        train_dir
        / "tflite_stream_state_internal_quant"
        / "stream_state_internal_quant.tflite"
    )
    if not candidate.exists():
        raise FileNotFoundError(
            f"Expected quantized streaming model at {candidate}, but it was not created."
        )
    return candidate


def start_http_server(serve_dir: Path) -> tuple[ThreadingHTTPServer, threading.Thread]:
    handler = partial(SimpleHTTPRequestHandler, directory=str(serve_dir))
    server = ThreadingHTTPServer(("0.0.0.0", 8080), handler)
    thread = threading.Thread(target=server.serve_forever, name="http-server", daemon=True)
    thread.start()
    logging.info("http server listening on 0.0.0.0:8080")
    return server, thread


def start_progress_logger(stop_event: threading.Event) -> threading.Thread:
    def _log_status() -> None:
        if stop_event.wait(60):
            return
        logging.info("still training...")
        while not stop_event.wait(300):
            logging.info("still training...")

    thread = threading.Thread(target=_log_status, name="progress-logger", daemon=True)
    thread.start()
    return thread


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a wake word model inside Docker")
    parser.add_argument(
        "-c",
        "--wakeword",
        required=True,
        help="Wake word phrase to synthesise and train",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help="Number of synthetic wake word samples to generate",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Sample generation batch size",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=DEFAULT_TRAINING_STEPS,
        help="Training steps per iteration",
    )
    args = parser.parse_args()

    configure_logging()

    wakeword = args.wakeword.strip()
    if not wakeword:
        parser.error("Wake word must not be empty")

    slug = slugify_phrase(wakeword)
    session_dir = DEFAULT_WORKDIR / slug
    serve_dir = session_dir / "serve"
    samples_dir = session_dir / "generated_samples"
    features_dir = session_dir / "generated_augmented_features"
    negatives_dir = session_dir / "negative_datasets"

    session_dir.mkdir(parents=True, exist_ok=True)
    serve_dir.mkdir(parents=True, exist_ok=True)

    server, server_thread = start_http_server(serve_dir)

    try:
        ensure_wakeword_samples(
            wakeword=wakeword,
            samples_dir=samples_dir,
            max_samples=args.max_samples,
            batch_size=args.sample_batch_size,
        )
        ensure_negative_datasets(negatives_dir)
        generate_positive_feature_sets(samples_dir, features_dir)
        config_path, train_dir = write_training_config(
            session_dir=session_dir, slug=slug, training_steps=args.training_steps
        )

        logging.info(
            "starting training for '%s' this will take a while", wakeword
        )
        start_time = time.time()
        stop_event = threading.Event()
        progress_thread = start_progress_logger(stop_event)

        try:
            run_training_process(config_path, workdir=session_dir)
        finally:
            stop_event.set()
            progress_thread.join(timeout=1)

        model_path = locate_tflite_model(train_dir)
        served_model = serve_dir / f"{slug}.tflite"
        shutil.copy2(model_path, served_model)

        duration = timedelta(seconds=int(time.time() - start_time))
        logging.info(
            "training complete for '%s'; download at http://0.0.0.0:8080/%s.tflite (took %s)",
            wakeword,
            slug,
            duration,
        )

        logging.info(
            "serving trained models from %s; press Ctrl+C to stop", serve_dir
        )

        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            logging.info("shutdown requested, stopping server")
    except Exception as exc:  # pragma: no cover - to aid manual diagnosis
        logging.exception("failed to train wake word model: %s", exc)
        raise SystemExit(1) from exc
    finally:
        server.shutdown()
        server_thread.join(timeout=5)


if __name__ == "__main__":
    main()
