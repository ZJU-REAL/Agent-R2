"""data-sythesis prompts/solver_prompt.py -- Prompt for the Solver Agent.

This Agent receives questions designed by the question_agent and completes
productivity tasks as the "solver". The conversation trajectory is used to
build fine-tuning datasets, so the model should produce high-quality, natural
problem-solving processes.
"""
from __future__ import annotations
from typing import Any, Dict, List


def build_solver_system_prompt(max_steps: int = 30, upload_skills: bool = True) -> str:
    """Build the system prompt for the Solver Agent.

    Args:
        max_steps: Maximum number of steps to execute
        upload_skills: Whether to upload skills to the sandbox

    Returns:
        System prompt string
    """
    # Skills section
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
- Do NOT embed large text content in a single terminal command (heredoc, python3 -c, echo, etc.). Long command strings will cause execution failures.
- Instead, write a Python script to /home/workspace/write_file.py using multiple small append calls (<=30 lines per call), then run it with `python3 /home/workspace/write_file.py`.

{skills_section}

You have at most {max_steps} steps to complete the task."""


def build_solver_user_prompt(
    question: Dict[str, Any],
    attachment_paths: List[Dict[str, str]],
) -> str:
    """Build the user prompt for the Solver Agent.

    Args:
        question: Question data containing title, description, deliverable, attachments, etc.
        attachment_paths: List of attachment paths in the sandbox, each containing local_path and sandbox_path

    Returns:
        User prompt string
    """
    title = question.get("title", "")
    description = question.get("description", "")
    deliverable = question.get("deliverable", {})

    # Build deliverable info
    deliv_type = deliverable.get("type", "")
    deliv_filename = deliverable.get("filename", "")

    # Build attachment info
    att_text = "\n".join([
        f"- {a.get('sandbox_path', '')}\n  {a.get('description', '')}"
        for a in attachment_paths
    ]) if attachment_paths else "No attachments"

    # Upload path note
    upload_path_section = """Input attachments have been uploaded to the /mnt/user-data/uploads/ directory."""

    return f"""Please complete the following work task.

[Task Title]
{title}

[Task Description]
{description}

[Deliverable]
- File type: {deliv_type}
- Filename: {deliv_filename}

[Input Attachments]
{att_text}
{upload_path_section}
Please first review the content of the input attachments, then complete the task as required. Save the output file to the /mnt/user-data/outputs/ directory."""
