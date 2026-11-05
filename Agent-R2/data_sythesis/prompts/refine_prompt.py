"""data-sythesis prompts/refine_prompt.py -- Prompt for the Refine Agent.

After the solver produces reference deliverables, this agent inspects them
against the rubric: revise unreasonable unmet items, and revise the
deliverable once when unmet items are reasonable.
"""
from __future__ import annotations
from typing import Any, Dict, List


def build_refine_system_prompt(max_steps: int = 20) -> str:
    """Build the system prompt for the Refine Agent."""
    return f"""You are a verifier for synthesized productivity tasks. The solver has produced reference deliverables. Your job is to check those deliverables against the rubric, then make at most one round of revisions.

**File System**
- `/mnt/user-data/uploads` — Task input attachments. Read directly from this path.
- `/home/workspace` — Scratch space for temp files and intermediate work. Not visible externally.
- `/mnt/user-data/outputs` — Current reference deliverables. Inspect them here. If you revise a deliverable, overwrite the files in this directory.

**Available Tools**
You have 4 tools: `web_fetch`, `web_search`, `terminal` and `view_image_text`. Use them to inspect attachments and deliverables (open files, render pages, check tables and figures).

**Your Core Tasks**
1. Inspect the input attachments and the solver deliverables in `/mnt/user-data/outputs`.
2. For each rubric item, decide whether the deliverable satisfies it.
3. For each unmet item:
   - If it is unsupported by the attachments, mismatched with the instruction, or otherwise unreasonable, revise or drop that rubric item.
   - If it is reasonable, keep the rubric item and revise the deliverable in `/mnt/user-data/outputs` so that the item is satisfied. Revise deliverables at most once.
4. Do not invent requirements that are not grounded in the instruction or attachments.

**Return Format**
After inspection (and any one-time deliverable revision), return ONLY the following JSON:

{{
    "rubric": [
        {{
            "criterion": "Specific verifiable scoring description",
            "category": "content|form",
            "points": 2,
            "source": "Source basis for the rubric and the approach needed to fulfill it"
        }}
    ],
    "unmet_items": [
        {{
            "criterion": "Original unmet criterion",
            "action": "revise_rubric|revise_deliverable",
            "reason": "Why the item was unmet and what you did"
        }}
    ],
    "deliverable_revised": true
}}

`rubric` is the final rubric after any revisions. `deliverable_revised` is true only if you modified files in `/mnt/user-data/outputs`.

You have at most {max_steps} steps. Return the JSON when your task is complete."""


def build_refine_user_prompt(
    question: Dict[str, Any],
    rubric: List[Dict[str, Any]],
    attachment_paths: List[Dict[str, str]],
    deliverable_paths: List[str],
) -> str:
    """Build the user prompt for the Refine Agent."""
    title = question.get("title", "")
    description = question.get("description", "")
    deliverable = question.get("deliverable", {})
    deliv_type = deliverable.get("type", "")
    deliv_filename = deliverable.get("filename", "")

    att_text = "\n".join([
        f"- {a.get('sandbox_path', '')}\n  {a.get('description', '')}"
        for a in attachment_paths
    ]) if attachment_paths else "No attachments"

    deliv_text = "\n".join(f"- {p}" for p in deliverable_paths) if deliverable_paths else "No deliverable files found in /mnt/user-data/outputs"

    rubric_lines: List[str] = []
    for i, item in enumerate(rubric, 1):
        criterion = item.get("criterion", "")
        category = item.get("category", "")
        points = item.get("points", item.get("score", item.get("max_score", 0)))
        source = item.get("source", "")
        extra = f" source={source}" if source else ""
        rubric_lines.append(f"{i}. [{category}|{points} pts] {criterion}{extra}")
    rubric_text = "\n".join(rubric_lines) if rubric_lines else "No rubric provided."

    return f"""Please verify the solver deliverables against the rubric and revise if needed.

[Task Title]
{title}

[Task Description]
{description}

[Deliverable]
- File type: {deliv_type}
- Filename: {deliv_filename}

[Input Attachments]
{att_text}

[Solver Deliverables] (in /mnt/user-data/outputs)
{deliv_text}

[Rubric]
{rubric_text}

Inspect the attachments and deliverables first. Then return the final JSON. If a reasonable unmet item requires a deliverable fix, revise the files in /mnt/user-data/outputs once before returning JSON."""
