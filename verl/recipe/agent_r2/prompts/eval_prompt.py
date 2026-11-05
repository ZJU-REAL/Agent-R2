"""data-sythesis prompts/eval_prompt.py -- Prompt for the Eval Agent.

This Agent receives the rubric generated during question design and the solver's
conversation trajectory, then scores the solver's performance item by item.
"""
from __future__ import annotations
from typing import Any, Dict, List


def build_eval_system_prompt(max_steps: int = 10) -> str:
    """Build the system prompt for the Eval Agent.

    Args:
        max_steps: Maximum number of LLM turns

    Returns:
        System prompt string
    """
    return f"""You are a strict, impartial evaluator. Your job is to score a solver's problem-solving trajectory against a provided rubric.

**Your task**
- Read the rubric carefully — each item has an id, a criterion, and a max score.
- Read the solver's conversation trajectory (a sequence of assistant messages and tool results).
- For each rubric item, decide how many points (0 ~ max_score) the solver earned, and provide a brief reasoning.

**Scoring principles**
- Be objective. Only award points when the criterion is clearly satisfied by evidence in the trajectory.
- Partial credit is allowed when the criterion is partially met.
- Do not infer or assume success beyond what is shown in the trajectory.

**Output format**
When you have finished your assessment, output ONLY a JSON code block with the following structure and nothing else:

```json
{{
  "evaluations": [
    {{
      "rubric_item_id": "<id>",
      "score": <int>,
      "reasoning": "<one or two sentences>"
    }}
  ]
}}
```

You have at most {max_steps} thinking steps before you must output the final JSON."""


def build_eval_user_prompt(
    rubric: List[Dict[str, Any]],
    solver_trajectory: List[Dict[str, Any]],
    question: Dict[str, Any] | None = None,
) -> str:
    """Build the user prompt for the Eval Agent.

    Args:
        rubric: List of rubric items, each with rubric_item_id, criterion, score
        solver_trajectory: The solver's conversation messages (role/content/tool_calls)
        question: Optional question dict for context (title, description, deliverable)

    Returns:
        User prompt string
    """
    # --- Rubric section ---
    rubric_lines: List[str] = []
    for item in rubric:
        item_id = item.get("rubric_item_id", item.get("id", ""))
        criterion = item.get("criterion", "")
        max_score = item.get("score", item.get("max_score", 0))
        rubric_lines.append(f"- [{item_id}] (max {max_score} pts) {criterion}")
    rubric_text = "\n".join(rubric_lines) if rubric_lines else "No rubric provided."

    # --- Question context (optional) ---
    question_section = ""
    if question:
        title = question.get("title", "")
        description = question.get("description", "")
        deliverable = question.get("deliverable", {})
        deliv_type = deliverable.get("type", "")
        deliv_filename = deliverable.get("filename", "")
        question_section = f"""
[Task Context]
Title: {title}
Description: {description}
Deliverable: {deliv_type} / {deliv_filename}
"""

    # --- Trajectory section ---
    trajectory_parts: List[str] = []
    for i, msg in enumerate(solver_trajectory):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            continue  # skip system prompt in trajectory display

        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            reasoning = msg.get("reasoning_content", "")
            parts = []
            if reasoning:
                parts.append(f"<thinking>\n{reasoning}\n</thinking>")
            if content:
                parts.append(content)
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    parts.append(
                        f"[Tool call] {fn.get('name', '')}({fn.get('arguments', '')})"
                    )
            trajectory_parts.append(f"### Turn {i} — assistant\n" + "\n".join(parts))

        elif role == "tool":
            trajectory_parts.append(
                f"### Turn {i} — tool result\n{content}"
            )

        elif role == "user":
            trajectory_parts.append(f"### Turn {i} — user\n{content}")

    trajectory_text = "\n\n".join(trajectory_parts) if trajectory_parts else "No trajectory provided."

    return f"""Please evaluate the solver's trajectory according to the rubric below.
{question_section}
[Rubric]
{rubric_text}

[Solver Trajectory]
{trajectory_text}

Score each rubric item based solely on evidence in the trajectory, then output the JSON result."""
