"""sythesis_test prompt.py -- Solver 和 Evaluator 的 prompt 模板。"""
from __future__ import annotations

from typing import Any, Dict, List


# ──────────────────────────────────────────────
# Solver prompts
# ──────────────────────────────────────────────

def build_solver_system_prompt(max_steps: int = 30, upload_skills: bool = True) -> str:
    """构建 Solver Agent 的系统 prompt。

    Args:
        max_steps: 最多执行的轮数
        upload_skills: 是否上传skills到沙盒
    """
    if upload_skills:
        skills_section = """
**File Operation Skills**
- docx: SKILL.md at /home/workspace/skills/docx/SKILL.md, scripts at /home/workspace/skills/docx/scripts/
- xlsx: SKILL.md at /home/workspace/skills/xlsx/SKILL.md, scripts at /home/workspace/skills/xlsx/scripts/
- pptx: SKILL.md at /home/workspace/skills/pptx/SKILL.md, scripts at /home/workspace/skills/pptx/scripts/
- pdf: SKILL.md at /home/workspace/skills/pdf/SKILL.md, scripts at /home/workspace/skills/pdf/scripts/

Read the SKILL.md content using terminal tool, then follow the guidelines. Use scripts from the scripts path if needed."""
    else:
        skills_section = ""


    return f"""You are a capable, resourceful AI Agent. You cut through ambiguity, plan before acting, and deliver high-quality results.

**File System**
- `/mnt/user-data/uploads` — User-uploaded files. Read directly from this path.
- `/home/workspace` — Scratch space for temp files and intermediate work. Not visible to the user.
- `/mnt/user-data/outputs` — Save final deliverables here before replying. Files not copied here are invisible to the user.

**Writing Files**
- Never put large text content inside a single terminal command (heredoc, python3 -c, echo, etc.) — long command strings will cause: `Tool terminal failed: Failed to exec command: exec_res`.

{skills_section}

You have at most {max_steps} steps to complete the task."""


def build_solver_user_prompt(
    task_prompt: str,
    reference_file_paths: List[str],
) -> str:
    """构建 Solver Agent 的用户 prompt。

    Args:
        task_prompt: benchmark 中的原始 prompt
        reference_file_paths: 参考文件在 sandbox 中的路径列表
        skill_info: skills信息字典
        upload_skills: 是否上传skills到沙盒
    """
    if reference_file_paths:
        ref_section = f"""
【Input Files】 (located in /mnt/user-data/uploads)
{chr(10).join(f"- {p}" for p in reference_file_paths)}
Please first inspect the input files, then complete the task according to the requirements."""
    else:
        ref_section = "\nNote: This task has no input files. Do NOT check or access /mnt/user-data/uploads/."

    return f"""Please complete the following work task.

【Task】
{task_prompt}
{ref_section}
Save the output file to the /mnt/user-data/outputs/ directory."""


# ──────────────────────────────────────────────
# Evaluator prompts
# ──────────────────────────────────────────────

def build_evaluator_system_prompt(max_steps: int = 20) -> str:
    """构建 Evaluator Agent 的系统 prompt。"""
    return f"""You are a strict task evaluator. Your job is to evaluate a submitted deliverable file against a detailed rubric and assign scores.

【Core Tasks】
1. Read the rubric carefully, understanding each scoring criterion
2. Use tools to inspect the submitted deliverable file
3. Compare the deliverable against each rubric criterion
4. Assign a score for each criterion (0 or the full score for that criterion)
5. Output a structured JSON evaluation result

【Work Style】
- Be thorough: use terminal to inspect file properties, content, and structure
- Be precise: check each criterion independently
- Be fair: only award points when the criterion is clearly met
- Be evidence-based: cite what you found in the file when awarding or denying points

【Available Tools】
- terminal: Execute commands in the sandbox (view file contents, inspect structure, etc.)
- view_image_text: Use AI vision to understand images in the sandbox

【Sandbox Working Directory】
- All files are located under /home/workspace/
- You may ONLY read/inspect files under /home/workspace/
- Do NOT access or modify any paths outside /home/workspace/
- Reference deliverables (ground truth) are placed at /home/workspace/reference_deliverable/

【Evaluation Rules】
- For each rubric item, assign either 0 or the full score (no partial scores)
- A criterion is met only if the evidence clearly supports it
- When in doubt, do not award the point
- Total score is the sum of all individual item scores

You have at most {max_steps} steps to complete the evaluation."""


def build_evaluator_user_prompt(
    rubric_pretty: str,
    rubric_items: List[Dict[str, Any]],
    deliverable_sandbox_path: str,
    reference_sandbox_paths: List[str],
) -> str:
    """构建 Evaluator Agent 的用户 prompt。

    Args:
        rubric_pretty: 人类可读的 rubric 文本
        rubric_items: rubric JSON 列表
        deliverable_sandbox_path: 待评估交付物在 sandbox 中的路径
        reference_sandbox_paths: 参考文件在 sandbox 中的路径
    """
    # 构建简洁的 rubric 列表
    rubric_list = []
    for i, item in enumerate(rubric_items, 1):
        rubric_list.append(
            f"  {i}. [ID: {item.get('rubric_item_id', '')}] [Score: {item.get('score', 0)}] {item.get('criterion', '')}"
        )
    rubric_text = "\n".join(rubric_list)
    total_max = sum(item.get("score", 0) for item in rubric_items)

    ref_text = "\n".join(
        f"- {p}" for p in reference_sandbox_paths
    ) if reference_sandbox_paths else "No reference files."

    return f"""Please evaluate the submitted deliverable file against the following rubric.

【Rubric】 (Total possible score: {total_max})
{rubric_text}

【Detailed Rubric】
{rubric_pretty}

【Submitted Deliverable】
{deliverable_sandbox_path}

【Reference Files】 (for comparison)
{ref_text}

Please inspect the deliverable file using the available tools and evaluate it against each rubric criterion.

After your evaluation, you MUST output a JSON block in the following format:
```json
{{
  "evaluations": [
    {{
      "rubric_item_id": "<rubric_item_id>",
      "score": <0 or full_score>,
      "reasoning": "<brief explanation>"
    }}
  ]
}}
```

IMPORTANT: Only include rubric_item_id, score, and reasoning in each evaluation item. Do NOT include criterion or max_score fields."""


# ──────────────────────────────────────────────
# Evaluator retry prompts
# ──────────────────────────────────────────────

def build_evaluator_json_retry_message() -> str:
    """构建 Evaluator JSON 解析失败时的重试提示。"""
    return """无法解析你返回的JSON格式,检查有没有转义错误,重新输出有效的JSON"""
