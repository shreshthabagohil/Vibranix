"""
preprocess.py
==============
Convert raw audio (.wav/m4a/...) into mel-spectrogram images sized (224 x 224 x 3).

Pipeline:
1) Load insect clips (Pest / Beneficial) from `data/raw/` if present, else fall back to:
   - `data/pest_indicators/`      -> Pest
   - `data/beneficial_indicators/` -> Beneficial
2) Load farm noise from `data/noise_samples/` if present, else fall back to:
   - `data/background_noise/` -> Noise
3) Augment: mix 20% noise (relative RMS) into every insect clip to simulate a farm environment.
4) Save 3-channel mel-spectrograms as `.npy` arrays and write a CSV manifest.

Outputs (default):
  data/processed/
    spectrograms/
      Pest/
      Beneficial/
      Noise/
    manifest.csv
    class_names.json
    preprocess_config.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import sys
import time

# #region agent log:env_info
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / ".cursor" / "debug-9f3473.log"

_RUN_ID = "run_import_debug"
_SESSION_ID = "9f3473"

def _agent_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": _SESSION_ID,
        "runId": _RUN_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

_agent_log(
    hypothesis_id="H2_wrong_python_env",
    location="scripts/preprocess.py:import_start",
    message="Python environment used for imports",
    data={"executable": sys.executable, "version": sys.version},
)
# #endregion

# #region agent log:missing_import
try:
    import librosa  # type: ignore
except ModuleNotFoundError as e:
    _agent_log(
        hypothesis_id="H1_deps_missing",
        location="scripts/preprocess.py:missing_librosa",
        message="Third-party dependency missing during import",
        data={"missing_module": getattr(e, "name", None)},
    )
    raise

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError as e:
    _agent_log(
        hypothesis_id="H1_deps_missing",
        location="scripts/preprocess.py:missing_numpy",
        message="Third-party dependency missing during import",
        data={"missing_module": getattr(e, "name", None)},
    )
    raise

try:
    import tensorflow as tf  # type: ignore
except ModuleNotFoundError as e:
    _agent_log(
        hypothesis_id="H1_deps_missing",
        location="scripts/preprocess.py:missing_tensorflow",
        message="Third-party dependency missing during import",
        data={"missing_module": getattr(e, "name", None)},
    )
    raise
# #endregion


DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PEST_INDICATORS_DIR = PROJECT_ROOT / "data" / "pest_indicators"
DEFAULT_BENEFICIAL_INDICATORS_DIR = PROJECT_ROOT / "data" / "beneficial_indicators"

DEFAULT_NOISE_SAMPLES_DIR = PROJECT_ROOT / "data" / "noise_samples"
DEFAULT_BACKGROUND_NOISE_DIR = PROJECT_ROOT / "data" / "background_noise"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

CLASS_NAMES = ["Pest", "Beneficial", "Noise"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Required by your model input spec.
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".aifc"}


def _first_existing_path(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def resolve_data_paths(raw_dir: Optional[str], noise_dir: Optional[str]) -> Tuple[Optional[Path], Path]:
    resolved_raw_dir: Optional[Path] = None
    if raw_dir:
        p = Path(raw_dir)
        if p.exists():
            resolved_raw_dir = p
    if resolved_raw_dir is None:
        resolved_raw_dir = DEFAULT_RAW_DIR if DEFAULT_RAW_DIR.exists() else None

    resolved_noise_dir: Optional[Path] = None
    if noise_dir:
        p = Path(noise_dir)
        if p.exists():
            resolved_noise_dir = p
    if resolved_noise_dir is None:
        resolved_noise_dir = _first_existing_path([DEFAULT_NOISE_SAMPLES_DIR, DEFAULT_BACKGROUND_NOISE_DIR])
    if resolved_noise_dir is None:
        raise FileNotFoundError(
            "Could not find noise directory. Expected `data/noise_samples/` or `data/background_noise/`."
        )

    return resolved_raw_dir, resolved_noise_dir


def list_audio_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    files.sort(key=lambda x: str(x))
    return files


def map_insect_folder_to_label(folder_name: str) -> Optional[str]:
    name = folder_name.strip().lower()
    if any(k in name for k in ["locust", "armyworm", "fall_armyworm", "fall armyworm"]):
        return "Pest"
    if "bee" in name or "honeybee" in name:
        return "Beneficial"
    return None


def normalize_mel_to_0_255(mel_db: np.ndarray) -> np.ndarray:
    # mel_db is log-scaled. Normalize per-sample for robust training.
    mel_db = mel_db.astype(np.float32)
    mel_db = mel_db - float(np.min(mel_db))
    denom = float(np.max(mel_db)) + 1e-8
    mel_norm = mel_db / denom  # [0, 1]
    mel_img = np.clip(mel_norm * 255.0, 0.0, 255.0)
    return mel_img.astype(np.float32)


def load_audio_fixed(
    audio_path: Path,
    sr: int,
    duration_s: float,
) -> np.ndarray:
    # librosa.load resamples to sr and converts to mono.
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    target_len = int(round(sr * duration_s))
    if y.shape[0] < target_len:
        y = np.pad(y, (0, target_len - y.shape[0]), mode="constant")
    else:
        y = y[:target_len]
    return y.astype(np.float32)


def tile_or_crop_to_match(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.shape[0] == target_len:
        return x
    if x.shape[0] > target_len:
        return x[:target_len]
    # Tile to reach target length.
    reps = int(np.ceil(target_len / x.shape[0]))
    y = np.tile(x, reps=reps)[:target_len]
    return y.astype(np.float32)


def mix_insect_with_noise(
    insect_audio: np.ndarray,
    noise_audio: np.ndarray,
    noise_mix_rms: float = 0.2,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Mix 20% noise "volume" into insect audio.
    Interpret "20%" as relative RMS contribution:
      desired_noise_rms = 0.2 * insect_rms
    """
    target_len = insect_audio.shape[0]
    noise_audio = tile_or_crop_to_match(noise_audio, target_len)

    insect_rms = float(np.sqrt(np.mean(insect_audio**2) + eps))
    noise_rms = float(np.sqrt(np.mean(noise_audio**2) + eps))
    desired_noise_rms = noise_mix_rms * insect_rms
    scale = desired_noise_rms / (noise_rms + eps)
    noise_scaled = noise_audio * scale

    mixed = insect_audio + noise_scaled
    # Normalize to avoid clipping artifacts dominating.
    peak = float(np.max(np.abs(mixed)) + eps)
    mixed = mixed / peak
    return mixed.astype(np.float32)


def compute_mel_spectrogram_image(
    audio: np.ndarray,
    sr: int,
    n_mels: int,
    fmax: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        fmax=fmax,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )  # (n_mels, t)
    mel_db = librosa.power_to_db(mel, ref=np.max)  # (n_mels, t)
    mel_img = normalize_mel_to_0_255(mel_db)  # (n_mels, t) in [0,255]

    # Resize to (224,224) and replicate to RGB channels.
    mel_img = mel_img[..., None]  # (n_mels, t, 1)
    mel_resized = tf.image.resize(tf.convert_to_tensor(mel_img), (IMG_HEIGHT, IMG_WIDTH), method="bilinear")
    mel_resized = mel_resized.numpy().astype(np.float32)  # (224,224,1)
    mel_resized = np.clip(mel_resized, 0.0, 255.0)

    mel_rgb = np.repeat(mel_resized, IMG_CHANNELS, axis=-1)  # (224,224,3)
    return mel_rgb.astype(np.float32)


def build_insect_sources(
    raw_dir: Optional[Path],
    max_files: Optional[int],
    seed: int,
) -> List[Tuple[Path, str]]:
    """
    Returns list of (audio_path, label_name) for insect clips.
    """
    rng = np.random.default_rng(seed)

    sources: List[Tuple[Path, str]] = []
    if raw_dir is not None and raw_dir.exists():
        # Expected structure under `data/raw/`:
        #   data/raw/locust/*.wav
        #   data/raw/fall_armyworm/*.wav
        #   data/raw/bee/*.wav
        for sub in raw_dir.iterdir():
            if not sub.is_dir():
                continue
            label = map_insect_folder_to_label(sub.name)
            if label is None:
                continue
            audio_files = list_audio_files(sub)
            for f in audio_files:
                sources.append((f, label))
    else:
        # Fallback to already-existing repo structure.
        pest_files = []
        if DEFAULT_PEST_INDICATORS_DIR.exists():
            pest_files = list_audio_files(DEFAULT_PEST_INDICATORS_DIR)
        beneficial_files = []
        if DEFAULT_BENEFICIAL_INDICATORS_DIR.exists():
            beneficial_files = list_audio_files(DEFAULT_BENEFICIAL_INDICATORS_DIR)

        sources.extend([(p, "Pest") for p in pest_files])
        sources.extend([(p, "Beneficial") for p in beneficial_files])

    sources.sort(key=lambda x: str(x[0]))
    if max_files is not None and len(sources) > max_files:
        indices = rng.choice(len(sources), size=max_files, replace=False)
        sources = [sources[i] for i in indices]
    return sources


def build_noise_sources(noise_dir: Path, max_files: Optional[int], seed: int) -> List[Path]:
    rng = np.random.default_rng(seed)
    files = list_audio_files(noise_dir)
    files.sort(key=lambda x: str(x))
    if max_files is not None and len(files) > max_files:
        indices = rng.choice(len(files), size=max_files, replace=False)
        files = [files[i] for i in indices]
    return files


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess audio into mel-spectrogram tensors.")
    parser.add_argument("--raw_dir", default=None, help="Path to `data/raw/` (optional override).")
    parser.add_argument("--noise_dir", default=None, help="Path to `data/noise_samples/` (optional override).")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate (Hz).")
    parser.add_argument("--duration", type=float, default=2.0, help="Fixed clip duration (seconds).")
    parser.add_argument("--n_mels", type=int, default=128, help="Mel bins (must be 128).")
    parser.add_argument("--fmax", type=int, default=8000, help="Max frequency for mel (must be 8000).")
    parser.add_argument("--n_fft", type=int, default=2048, help="FFT size for mel.")
    parser.add_argument("--hop_length", type=int, default=256, help="Hop length for mel.")
    parser.add_argument("--noise_mix_ratio", type=float, default=0.2, help="Noise RMS contribution ratio.")
    parser.add_argument("--insect_max_files", type=int, default=None, help="Limit number of insect audio files.")
    parser.add_argument("--noise_max_files", type=int, default=None, help="Limit number of noise audio files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing spectrogram .npy files.")
    args = parser.parse_args()

    raw_dir, noise_dir = resolve_data_paths(args.raw_dir, args.noise_dir)
    output_dir = Path(args.output_dir)
    spectrograms_dir = output_dir / "spectrograms"
    ensure_dir(spectrograms_dir)

    for cls in CLASS_NAMES:
        ensure_dir(spectrograms_dir / cls)

    insect_sources = build_insect_sources(raw_dir=raw_dir, max_files=args.insect_max_files, seed=args.seed)
    noise_sources = build_noise_sources(noise_dir=noise_dir, max_files=args.noise_max_files, seed=args.seed)
    if not insect_sources:
        raise RuntimeError("No insect audio sources found. Check your `data/raw` or fallback directories.")
    if not noise_sources:
        raise RuntimeError(f"No noise audio found in {noise_dir}.")

    rng = np.random.default_rng(args.seed)

    manifest_path = output_dir / "manifest.csv"
    class_names_path = output_dir / "class_names.json"
    config_path = output_dir / "preprocess_config.json"

    config = {
        "sample_rate": args.sample_rate,
        "duration": args.duration,
        "n_mels": args.n_mels,
        "fmax": args.fmax,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "noise_mix_ratio": args.noise_mix_ratio,
        "img_size": [IMG_HEIGHT, IMG_WIDTH],
        "img_channels": IMG_CHANNELS,
        "raw_dir_used": str(raw_dir) if raw_dir is not None else None,
        "noise_dir_used": str(noise_dir),
    }
    ensure_dir(output_dir)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    class_names_path.write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")

    # Generate spectrograms and manifest.
    rows: List[Dict[str, str]] = []

    def spectrogram_out_path(audio_path: Path) -> Path:
        # Stable unique id per source file location.
        rel = None
        try:
            rel = audio_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = audio_path.name
        uid = hashlib.md5(str(rel).encode("utf-8")).hexdigest()[:10]
        return spectrograms_dir / label / f"{audio_path.stem}_{uid}.npy"

    # Insect -> apply noise mix augmentation
    insect_count = 0
    for audio_path, label in insect_sources:
        out_path = spectrogram_out_path(audio_path)
        if out_path.exists() and not args.overwrite:
            insect_count += 1
            rows.append({"path": str(out_path), "label": label, "label_idx": str(CLASS_TO_INDEX[label])})
            continue

        insect_audio = load_audio_fixed(audio_path, sr=args.sample_rate, duration_s=args.duration)
        noise_path = rng.choice(noise_sources)
        noise_audio = load_audio_fixed(noise_path, sr=args.sample_rate, duration_s=args.duration)

        mixed_audio = mix_insect_with_noise(
            insect_audio=insect_audio,
            noise_audio=noise_audio,
            noise_mix_rms=args.noise_mix_ratio,
        )

        mel_rgb = compute_mel_spectrogram_image(
            audio=mixed_audio,
            sr=args.sample_rate,
            n_mels=args.n_mels,
            fmax=args.fmax,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
        )

        np.save(out_path, mel_rgb.astype(np.float32))
        rows.append({"path": str(out_path), "label": label, "label_idx": str(CLASS_TO_INDEX[label])})
        insect_count += 1
        if insect_count % 50 == 0:
            print(f"[preprocess] Insect processed: {insect_count}/{len(insect_sources)}")

    # Noise class -> use noise clips directly (no augmentation needed)
    noise_count = 0
    label = "Noise"
    for audio_path in noise_sources:
        out_path = spectrogram_out_path(audio_path)
        if out_path.exists() and not args.overwrite:
            noise_count += 1
            rows.append({"path": str(out_path), "label": label, "label_idx": str(CLASS_TO_INDEX[label])})
            continue

        noise_audio = load_audio_fixed(audio_path, sr=args.sample_rate, duration_s=args.duration)
        mel_rgb = compute_mel_spectrogram_image(
            audio=noise_audio,
            sr=args.sample_rate,
            n_mels=args.n_mels,
            fmax=args.fmax,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
        )
        np.save(out_path, mel_rgb.astype(np.float32))
        rows.append({"path": str(out_path), "label": label, "label_idx": str(CLASS_TO_INDEX[label])})
        noise_count += 1
        if noise_count % 50 == 0:
            print(f"[preprocess] Noise processed: {noise_count}/{len(noise_sources)}")

    # Write manifest
    rows.sort(key=lambda r: r["path"])
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "label_idx"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[preprocess] Done. Wrote {len(rows)} samples to: {output_dir}")
    print(f"[preprocess] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

