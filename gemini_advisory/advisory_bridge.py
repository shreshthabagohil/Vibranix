"""
advisory_bridge.py
==================
Gemini 1.5 Flash bridge for Vibranix.

Reads:
  gemini_advisory/pest_knowledge.json

Exposes:
  get_advisory(detection: dict) -> dict

Returns STRICT JSON with exactly:
  - Indian_Name
  - Organic_Control
  - Risk_Level

Env:
  - GEMINI_API_KEY
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT_ROOT / "gemini_advisory" / "pest_knowledge.json"

# Preferred model order. Some accounts/APIs do not expose older v1beta model ids.
MODEL_CANDIDATES = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
]
CLASS_NAMES = ["Pest", "Beneficial", "Noise"]
EXPECTED_KEYS = ("Indian_Name", "Organic_Control", "Risk_Level")

def _load_knowledge() -> Optional[Any]:
    if not KNOWLEDGE_PATH.exists():
        return None
    return json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))

def _normalize_detection(detection: Dict[str, Any]) -> Tuple[str, float]:
    conf = float(detection.get("confidence", 0.85))
    label = detection.get("label", "Pest").strip().capitalize()
    return label, conf

def get_advisory(detection: Dict[str, Any]) -> Dict[str, str]:
    # Force .env values to override any stale shell-exported vars.
    load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing API Key in .env")

    client = genai.Client(api_key=api_key)
    label, conf = _normalize_detection(detection)
    knowledge = _load_knowledge()

    prompt = (
        f"You are an expert Indian agricultural advisor. Provide advice for a detected '{label}' "
        f"with {conf:.2f} confidence. Local database context: {json.dumps(knowledge)}. "
        "Return the response in strict JSON format."
    )

    last_error: Exception | None = None
    resp = None
    for model_name in MODEL_CANDIDATES:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "Indian_Name": {"type": "STRING"},
                            "Organic_Control": {"type": "STRING"},
                            "Risk_Level": {"type": "STRING"}
                        },
                        "required": ["Indian_Name", "Organic_Control", "Risk_Level"]
                    }
                )
            )
            break
        except Exception as e:
            last_error = e
            continue

    if resp is None:
        raise RuntimeError(f"No compatible Gemini model available. Last error: {last_error}")
    
    # Extract just the JSON part
    return json.loads(resp.text)

if __name__ == "__main__":
    # Test it right now!
    print("🚀 Testing Vibranix Advisory Bridge...")
    test_data = {"label": "Locust", "confidence": 0.92}
    try:
        result = get_advisory(test_data)
        print("\n✅ API Response Received:")
        print(json.dumps(result, indent=4))
    except Exception as e:
        print(f"❌ Error: {e}")