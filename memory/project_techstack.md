---
name: Vibranix Tech Stack
description: Eco-Acoustic Pest Radar hackathon project tech stack and architecture requirements
type: project
---

24-hour hackathon project "Eco-Acoustic Pest Radar" (Vibranix).

**Tech Stack:**
- Frontend: Flutter (mobile-first, mic access)
- Backend: Firebase (Firestore & Auth)
- Edge AI: TensorFlow Lite (offline on-device inference)
- Training: Edge Impulse (audio → Mel-Spectrogram → CNN)
- Denoising: SoX (Sound eXchange) — DSP layer for farm noise removal
- Advisory: Gemini 1.5 Flash API
- Map: Google Maps API + Geoflutterfire
- TFLite Flutter Plugin for deployment

**Why:** Rural/offline-first, must run on mid-range Android/iOS.

**How to apply:** All scripts must produce Edge Impulse-compatible formats. Denoising uses SoX, not custom Python DSP. Model export must be .tflite.
