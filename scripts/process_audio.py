"""
process_audio.py
================
Full pre-processing pipeline for Vibranix Acoustic Radar.

What it does:
- Loads insect clips from `data/raw/` (expected) or falls back to existing repo folders:
    - `data/pest_indicators/`       -> Pest
    - `data/beneficial_indicators/` -> Beneficial
- Loads farm noise from `data/noise_samples/` (expected) or falls back to:
    - `data/background_noise/` -> Noise
- Augments: mixes 20% noise (relative RMS) into EVERY insect clip
- Converts each clip to a mel-spectrogram image tensor (224 x 224 x 3)
- Saves `.npy` tensors into `data/processed/` and writes `manifest.csv`
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

FALLBACK_PEST_DIR = PROJECT_ROOT / "data" / "pest_indicators"
FALLBACK_BENEFICIAL_DIR = PROJECT_ROOT / "data" / "beneficial_indicators"
FALLBACK_NOISE_DIR = PROJECT_ROOT / "data" / "background_noise"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

CLASS_NAMES = ["Pest", "Beneficial", "Noise"]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASS_NAMES)}

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".aifc"}


def _first_existing_path(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def list_audio_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            out.append(p)
    out.sort(key=lambda x: str(x))
    return out


def map_insect_folder_to_label(folder_name: str) -> Optional[str]:
    name = folder_name.strip().lower()
    if any(k in name for k in ["locust", "armyworm", "fall_armyworm", "fall armyworm"]):
        return "Pest"
    if "bee" in name or "honeybee" in name:
        return "Beneficial"
    return None


def load_audio_fixed(path: Path, sr: int, duration_s: float) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
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
        return x[:target_len].astype(np.float32)
    reps = int(np.ceil(target_len / x.shape[0]))
    return np.tile(x, reps=reps)[:target_len].astype(np.float32)


def mix_with_noise(insect: np.ndarray, noise: np.ndarray, noise_rms_ratio: float = 0.2, eps: float = 1e-8) -> np.ndarray:
    target_len = insect.shape[0]
    noise = tile_or_crop_to_match(noise, target_len)

    insect_rms = float(np.sqrt(np.mean(insect**2) + eps))
    noise_rms = float(np.sqrt(np.mean(noise**2) + eps))
    desired_noise_rms = noise_rms_ratio * insect_rms
    scale = desired_noise_rms / (noise_rms + eps)

    mixed = insect + (noise * scale)
    peak = float(np.max(np.abs(mixed)) + eps)
    return (mixed / peak).astype(np.float32)


def normalize_mel_to_0_255(mel_db: np.ndarray) -> np.ndarray:
    mel_db = mel_db.astype(np.float32)
    mel_db = mel_db - float(np.min(mel_db))
    denom = float(np.max(mel_db)) + 1e-8
    mel_norm = mel_db / denom
    return np.clip(mel_norm * 255.0, 0.0, 255.0).astype(np.float32)


def mel_image(audio: np.ndarray, sr: int, n_mels: int, fmax: int, n_fft: int, hop_length: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        fmax=fmax,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_img = normalize_mel_to_0_255(mel_db)[..., None]  # (n_mels, t, 1)
    resized = tf.image.resize(tf.convert_to_tensor(mel_img), (IMG_HEIGHT, IMG_WIDTH), method="bilinear").numpy()
    resized = np.clip(resized.astype(np.float32), 0.0, 255.0)
    return np.repeat(resized, IMG_CHANNELS, axis=-1).astype(np.float32)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_insect_sources(raw_dir: Path) -> List[Tuple[Path, str]]:
    sources: List[Tuple[Path, str]] = []
    if raw_dir.exists():
        for sub in raw_dir.iterdir():
            if not sub.is_dir():
                continue
            label = map_insect_folder_to_label(sub.name)
            if label is None:
                continue
            for f in list_audio_files(sub):
                sources.append((f, label))
    # If `data/raw` exists but is empty/unlabeled, fall back to existing repo folders.
    if not sources:
        if FALLBACK_PEST_DIR.exists():
            sources.extend([(p, "Pest") for p in list_audio_files(FALLBACK_PEST_DIR)])
        if FALLBACK_BENEFICIAL_DIR.exists():
            sources.extend([(p, "Beneficial") for p in list_audio_files(FALLBACK_BENEFICIAL_DIR)])

    sources.sort(key=lambda x: str(x[0]))
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix noise + convert audio to 224x224x3 mel-spectrogram tensors.")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--fmax", type=int, default=8000)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--noise_mix_ratio", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Ensure expected folder structure exists (even if empty)
    ensure_dir(RAW_DIR)

    # Per requirement: ALWAYS use `data/background_noise/` for augmentation mixing.
    noise_dir = FALLBACK_NOISE_DIR
    if not noise_dir.exists():
        raise FileNotFoundError("Missing `data/background_noise/` (required for noise augmentation).")

    insect_sources = build_insect_sources(RAW_DIR)
    if not insect_sources:
        raise RuntimeError("No insect audio found. Populate data/raw or use existing fallback folders.")

    noise_sources = list_audio_files(noise_dir)
    if not noise_sources:
        raise RuntimeError(f"No noise audio found in {noise_dir}.")

    out_root = OUTPUT_DIR
    spec_root = out_root / "spectrograms"
    ensure_dir(spec_root)
    for c in CLASS_NAMES:
        ensure_dir(spec_root / c)

    rng = np.random.default_rng(42)
    rows: List[Dict[str, str]] = []

    def out_path_for(audio_path: Path, label: str) -> Path:
        try:
            rel = audio_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = audio_path.name
        uid = hashlib.md5(str(rel).encode("utf-8")).hexdigest()[:10]
        return spec_root / label / f"{audio_path.stem}_{uid}.npy"

    # Insects (with augmentation)
    for audio_path, label in insect_sources:
        out_path = out_path_for(audio_path, label)
        if out_path.exists() and not args.overwrite:
            rows.append({"path": str(out_path), "label": label, "label_idx": str(CLASS_TO_INDEX[label])})
            continue

        insect = load_audio_fixed(audio_path, args.sample_rate, args.duration)
        noise_path = rng.choice(noise_sources)
        noise = load_audio_fixed(Path(noise_path), args.sample_rate, args.duration)
        mixed = mix_with_noise(insect, noise, noise_rms_ratio=args.noise_mix_ratio)
        img = mel_image(mixed, args.sample_rate, args.n_mels, args.fmax, args.n_fft, args.hop_length)
        np.save(out_path, img.astype(np.float32))
        rows.append({"path": str(out_path), "label": label, "label_idx": str(CLASS_TO_INDEX[label])})

    # Noise class (noise as-is)
    for audio_path in noise_sources:
        label = "Noise"
        out_path = out_path_for(audio_path, label)
        if out_path.exists() and not args.overwrite:
            rows.append({"path": str(out_path), "label": label, "label_idx": str(CLASS_TO_INDEX[label])})
            continue

        noise = load_audio_fixed(audio_path, args.sample_rate, args.duration)
        img = mel_image(noise, args.sample_rate, args.n_mels, args.fmax, args.n_fft, args.hop_length)
        np.save(out_path, img.astype(np.float32))
        rows.append({"path": str(out_path), "label": label, "label_idx": str(CLASS_TO_INDEX[label])})

    ensure_dir(out_root)
    (out_root / "class_names.json").write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")

    rows.sort(key=lambda r: r["path"])
    with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "label_idx"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[process_audio] Wrote {len(rows)} samples to {out_root}")
    print(f"[process_audio] Manifest: {out_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()

