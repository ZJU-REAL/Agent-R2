"""officeqa eval.py -- LLM-as-judge 评估器：把 ground_truth 和 predicted 给模型打分。"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from llm.router import call_model

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert judge evaluating whether a predicted answer matches the ground truth for U.S. Treasury financial questions.

Your task: determine if the predicted answer conveys the same value as the ground truth.

Rules:
- Numbers with different formatting but same value are correct (e.g. "2,602" == "2602" == "$2,602 million" if GT is "2,602" in millions)
- Ignore units/currency symbols if the numeric value matches the scale implied by the ground truth
- For percentage answers, "1608.80%" == "1608.8%" == "1608.80 percent"
- For text answers, minor differences in phrasing are acceptable if the meaning is the same
- If the predicted answer contains the correct value plus extra irrelevant text, it is still correct
- Return ONLY valid JSON, no other text

Output format:
{"correct": true/false, "reason": "brief explanation"}"""

_USER_TEMPLATE = """Ground truth: {ground_truth}
Predicted: {predicted}

Is the predicted answer correct?"""


async def run_evaluator(
    *,
    ground_truth: str,
    predicted: str,
    sk: str,
    api_url: str,
    model_id: str = "qwen3.6-plus",
    family: str = "qwen",
) -> Dict[str, Any]:
    if not predicted:
        return {"accuracy": 0.0, "reason": "empty prediction", "status": "scored"}

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(
            ground_truth=ground_truth,
            predicted=predicted,
        )},
    ]

    try:
        resp = await call_model(
            family=family,
            model_id=model_id,
            sk=sk,
            messages=messages,
            tools=None,
            api_url=api_url,
        )
    except Exception as e:
        logger.warning(f"[eval] LLM call failed: {e}")
        return {"accuracy": 0.0, "reason": f"llm error: {e}", "status": "error"}

    content = resp.get("content", "") or ""
    # strip <think>...</think> if present
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    try:
        # extract JSON even if wrapped in markdown code block
        m = re.search(r"\{.*\}", content, re.DOTALL)
        data = json.loads(m.group() if m else content)
        correct = bool(data.get("correct", False))
        reason = data.get("reason", "")
    except Exception as e:
        logger.warning(f"[eval] JSON parse failed: {e}, raw: {content!r}")
        return {"accuracy": 0.0, "reason": f"parse error: {content[:200]}", "status": "error"}

    return {
        "accuracy": 1.0 if correct else 0.0,
        "reason": reason,
        "status": "scored",
    }
