"""
advisory.py
============
Gemini advisory bridge.

Requirement:
- Provide a function that takes a detection result and queries Gemini 1.5 Flash.
- Return a Strict JSON response with exactly:
    - Indian_Name
    - Organic_Control
    - Risk_Level

Gemini configuration:
- Uses `GEMINI_API_KEY` from environment variables.

Optional local knowledge:
- If `gemini_advisory/pest_knowledge.json` exists, its content is included as context.
  (The repo currently may or may not have this file; this script is tolerant.)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        # Debug logging must never break the actual pipeline.
        pass

_agent_log(
    hypothesis_id="H2_wrong_python_env",
    location="scripts/advisory.py:import_start",
    message="Python environment used for imports",
    data={"executable": sys.executable, "version": sys.version},
)
# #endregion

# #region agent log:missing_import
try:
    import google.generativeai as genai  # type: ignore
except ModuleNotFoundError as e:
    _agent_log(
        hypothesis_id="H1_deps_missing",
        location="scripts/advisory.py:missing_google_generativeai",
        message="Third-party dependency missing during import",
        data={"missing_module": getattr(e, "name", None)},
    )
    raise
# #endregion


KNOWLEDGE_PATH = PROJECT_ROOT / "gemini_advisory" / "pest_knowledge.json"

GEMINI_MODEL = "gemini-1.5-flash"
EXPECTED_KEYS = ("Indian_Name", "Organic_Control", "Risk_Level")
RISK_ALLOWED = ("Low", "Moderate", "High")

CLASS_NAMES = ["Pest", "Beneficial", "Noise"]


def _load_local_knowledge() -> Optional[Any]:
    if not KNOWLEDGE_PATH.exists():
        return None
    try:
        return json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def normalize_detection_result(detection_result: Dict[str, Any]) -> Tuple[str, float]:
    """
    Returns (class_label, confidence).

    Supported input shapes:
      - {"class_index": 0, "confidence": 0.91}
      - {"label": "Pest", "confidence": 0.91}
      - {"predicted_label": "Pest", "confidence": 0.91}
      - {"class": "Pest", "confidence": 0.91}
    """
    confidence = float(detection_result.get("confidence", detection_result.get("score", 0.0)) or 0.0)

    class_label = (
        detection_result.get("label")
        or detection_result.get("predicted_label")
        or detection_result.get("class")
        or None
    )
    if class_label is None and "class_index" in detection_result:
        idx = int(detection_result["class_index"])
        if 0 <= idx < len(CLASS_NAMES):
            class_label = CLASS_NAMES[idx]

    if not class_label:
        # Last resort: treat unknown as Noise.
        class_label = "Noise"

    # Normalize label casing.
    class_label = str(class_label).strip().capitalize()
    if class_label not in CLASS_NAMES:
        class_label = "Noise"

    return class_label, confidence


def _extract_strict_json(text: str) -> Dict[str, Any]:
    """
    Best-effort extraction of a single JSON object from Gemini output.
    Then validates required keys.
    """
    text = text.strip()
    # Remove common markdown fences.
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    # If Gemini still wrapped extra text, extract the first JSON object.
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        text = m.group(0)

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Gemini returned non-object JSON.")

    missing = [k for k in EXPECTED_KEYS if k not in data]
    if missing:
        raise ValueError(f"Gemini JSON missing keys: {missing}")

    # Keep only required keys (strict response).
    out = {k: data[k] for k in EXPECTED_KEYS}

    # Normalize risk value if needed.
    risk = str(out["Risk_Level"]).strip()
    # Accept a few variants.
    lowered = risk.lower()
    if "low" in lowered:
        out["Risk_Level"] = "Low"
    elif "mod" in lowered:
        out["Risk_Level"] = "Moderate"
    elif "high" in lowered:
        out["Risk_Level"] = "High"
    else:
        if risk in RISK_ALLOWED:
            out["Risk_Level"] = risk
        else:
            out["Risk_Level"] = "Moderate"

    # Ensure string fields.
    out["Indian_Name"] = str(out["Indian_Name"]).strip()
    out["Organic_Control"] = str(out["Organic_Control"]).strip()
    return out


def get_gemini_advisory(detection_result: Dict[str, Any]) -> Dict[str, str]:
    """
    Queries Gemini 1.5 Flash and returns strict JSON:
      { "Indian_Name": ..., "Organic_Control": ..., "Risk_Level": ... }
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GEMINI_API_KEY environment variable.")

    class_label, confidence = normalize_detection_result(detection_result)
    knowledge = _load_local_knowledge()

    # Keep the prompt deterministic and strict.
    prompt_parts = [
        "You are an expert Indian agricultural entomologist and advisor.",
        "Return ONLY valid JSON (no markdown, no extra text).",
        "JSON schema (exact keys, exact spelling):",
        '{"Indian_Name":"...", "Organic_Control":"...", "Risk_Level":"Low|Moderate|High"}',
        "",
        "Detection result:",
        f"- Class: {class_label}",
        f"- Confidence: {confidence:.3f}",
        "",
        "Decision rules:",
        "- If Class is 'Pest' and Confidence >= 0.5: Risk_Level should be 'High' or 'Moderate' and Organic_Control must include organic measures suitable for common Indian field conditions.",
        "- If Class is 'Beneficial': Risk_Level should be 'Low' and Organic_Control should recommend preserving beneficial insects and avoiding harmful pesticides.",
        "- If Class is 'Noise' or Confidence < 0.5: Risk_Level should be 'Low' and Organic_Control should recommend no action (or basic sanity check).",
        "",
    ]

    if knowledge is not None:
        prompt_parts.append("Local pest remedy knowledge (use it when relevant):")
        prompt_parts.append(json.dumps(knowledge, ensure_ascii=False))
        prompt_parts.append("")

    prompt_parts.append("Now produce the strict JSON response.")
    prompt = "\n".join(prompt_parts)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    generation_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "response_mime_type": "application/json",
    }

    response = model.generate_content(prompt, generation_config=generation_config)
    text = getattr(response, "text", None) or str(response)
    return _extract_strict_json(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Get Gemini advisory from a detection result.")
    parser.add_argument("--class_label", default="Pest", choices=CLASS_NAMES)
    parser.add_argument("--confidence", type=float, default=0.9)
    args = parser.parse_args()

    result = get_gemini_advisory({"label": args.class_label, "confidence": args.confidence})
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

