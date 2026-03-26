"""
Step 1: Dataset Preparation for Eco-Acoustic Pest Radar
========================================================
Generates synthetic insect wingbeat audio based on published entomological data.

Wingbeat frequencies (from literature):
- Locust (Schistocerca gregaria): ~20-30 Hz fundamental
- Fall Armyworm (Spodoptera frugiperda): ~40-60 Hz fundamental
- Honeybee (Apis mellifera): ~200-250 Hz fundamental

Each synthetic sample includes:
- Fundamental frequency + harmonics
- Amplitude modulation (natural wingbeat variation)
- Light background noise for realism
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse


# --- Configuration ---
SAMPLE_RATE = 16000  # 16kHz — sufficient for insect audio, keeps model small
DURATION = 2.0       # 2 seconds per clip
NUM_SAMPLES_PER_CLASS = 200  # samples per insect class

INSECT_PROFILES = {
    "locust": {
        "freq_range": (20, 30),
        "harmonics": [2, 3, 4, 5],
        "harmonic_decay": 0.6,
        "am_rate": (3, 6),        # amplitude modulation rate (Hz)
        "am_depth": 0.4,
        "noise_level": 0.02,
    },
    "fall_armyworm": {
        "freq_range": (40, 60),
        "harmonics": [2, 3, 4],
        "harmonic_decay": 0.5,
        "am_rate": (5, 10),
        "am_depth": 0.3,
        "noise_level": 0.02,
    },
    "bee": {
        "freq_range": (200, 250),
        "harmonics": [2, 3],
        "harmonic_decay": 0.4,
        "am_rate": (2, 5),
        "am_depth": 0.2,
        "noise_level": 0.02,
    },
}


def generate_wingbeat(profile: dict, sr: int, duration: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a single synthetic wingbeat audio clip."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Random fundamental frequency within range
    f0 = rng.uniform(*profile["freq_range"])

    # Fundamental tone
    signal = np.sin(2 * np.pi * f0 * t)

    # Add harmonics with decay
    for i, h in enumerate(profile["harmonics"]):
        amp = profile["harmonic_decay"] ** (i + 1)
        # Slight detuning for realism
        detune = rng.uniform(-0.5, 0.5)
        signal += amp * np.sin(2 * np.pi * (f0 * h + detune) * t)

    # Amplitude modulation (simulates wingbeat rhythm)
    am_rate = rng.uniform(*profile["am_rate"])
    am_phase = rng.uniform(0, 2 * np.pi)
    am = 1.0 + profile["am_depth"] * np.sin(2 * np.pi * am_rate * t + am_phase)
    signal *= am

    # Random amplitude envelope (fade in/out)
    fade_in = rng.uniform(0.05, 0.2)
    fade_out = rng.uniform(0.05, 0.2)
    fade_in_samples = int(fade_in * sr)
    fade_out_samples = int(fade_out * sr)
    envelope = np.ones_like(signal)
    envelope[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)
    envelope[-fade_out_samples:] = np.linspace(1, 0, fade_out_samples)
    signal *= envelope

    # Add light background noise
    noise = rng.normal(0, profile["noise_level"], len(t))
    signal += noise

    # Normalize to [-1, 1]
    signal = signal / (np.abs(signal).max() + 1e-8)

    # Random gain variation
    gain = rng.uniform(0.5, 1.0)
    signal *= gain

    return signal.astype(np.float32)


def generate_dataset(output_dir: str, num_samples: int = NUM_SAMPLES_PER_CLASS, seed: int = 42):
    """Generate the full synthetic dataset."""
    rng = np.random.default_rng(seed)
    output_path = Path(output_dir)

    total = 0
    for insect_name, profile in INSECT_PROFILES.items():
        class_dir = output_path / insect_name
        class_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {num_samples} samples for '{insect_name}'...")
        for i in range(num_samples):
            audio = generate_wingbeat(profile, SAMPLE_RATE, DURATION, rng)
            filepath = class_dir / f"{insect_name}_{i:04d}.wav"
            sf.write(str(filepath), audio, SAMPLE_RATE)
            total += 1

        print(f"  -> Saved to {class_dir}/")

    print(f"\nDone! Generated {total} total samples in '{output_dir}'")
    print(f"Classes: {list(INSECT_PROFILES.keys())}")
    print(f"Sample rate: {SAMPLE_RATE} Hz | Duration: {DURATION}s each")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic insect wingbeat dataset")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES_PER_CLASS, help="Samples per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_dataset(args.output, args.samples, args.seed)
