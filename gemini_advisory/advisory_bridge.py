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
import google.generativeai as genai


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT_ROOT / "gemini_advisory" / "pest_knowledge.json"

MODEL_NAME = "gemini-1.5-flash"
CLASS_NAMES = ["Pest", "Beneficial", "Noise"]
EXPECTED_KEYS = ("Indian_Name", "Organic_Control", "Risk_Level")


def _load_knowledge() -> Optional[Any]:
    if not KNOWLEDGE_PATH.exists():
        return None
    return json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))


def _normalize_detection(detection: Dict[str, Any]) -> Tuple[str, float]:
    conf = float(detection.get("confidence", detection.get("score", 0.0)) or 0.0)
    label = detection.get("label") or detection.get("predicted_label") or detection.get("class")
    if label is None and "class_index" in detection:
        idx = int(detection["class_index"])
        if 0 <= idx < len(CLASS_NAMES):
            label = CLASS_NAMES[idx]
    label = (str(label).strip().capitalize() if label else "Noise")
    if label not in CLASS_NAMES:
        label = "Noise"
    return label, conf


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        text = m.group(0)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Gemini returned non-object JSON.")
    for k in EXPECTED_KEYS:
        if k not in data:
            raise ValueError(f"Gemini JSON missing key: {k}")
    return {k: str(data[k]).strip() for k in EXPECTED_KEYS}


def get_advisory(detection: Dict[str, Any]) -> Dict[str, str]:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GEMINI_API_KEY")

    label, conf = _normalize_detection(detection)
    knowledge = _load_knowledge()

    prompt = "\n".join(
        [
            "You are an expert Indian agricultural advisor.",
            "Return ONLY valid JSON (no markdown, no extra text).",
            'Schema: {"Indian_Name":"...", "Organic_Control":"...", "Risk_Level":"Low|Moderate|High"}',
            "",
            f"Detection: class={label}, confidence={conf:.3f}",
            "",
            "Rules:",
            "- If class is Pest and confidence >= 0.5: provide organic controls appropriate for Indian farms; risk is Moderate/High.",
            "- If class is Beneficial: risk Low; advise protecting beneficial insects.",
            "- If class is Noise OR confidence < 0.5: risk Low; advise no action.",
            "",
            "Local knowledge JSON (use when relevant):",
            json.dumps(knowledge, ensure_ascii=False) if knowledge is not None else "{}",
            "",
            "Now output the strict JSON.",
        ]
    )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(
        prompt,
        generation_config={"temperature": 0.2, "top_p": 0.9, "response_mime_type": "application/json"},
    )
    return _extract_json(getattr(resp, "text", "") or str(resp))

