"""data-sythesis prompts/question_prompt.py -- Prompt for the Question Design Agent.

This Agent is responsible for designing specific productivity questions based on
the scenario, workflow, and retrieved files, and generating rubric scoring criteria.
Questions must have attachments.
"""
from __future__ import annotations
from typing import Any, Dict, List


def build_question_system_prompt(max_steps: int = 30) -> str:
    """Build the system prompt for the Question Design Agent.

    Args:
        max_steps: Maximum number of steps to execute

    Returns:
        System prompt string
    """
    return f"""You are a professional productivity task question designer. Based on workflow scenarios and retrieved files, design high-quality productivity questions and generate rubric scoring criteria.

**File System**
- `/mnt/user-data/uploads` — Relevant files have been uploaded here. Read directly from this path.
  - Each file may have a companion **preprocessed folder** (same name as the file, without extension) containing `content.md` and extracted images. These are provided **only to help you understand the file content** — do NOT treat them as attachments for the question.
- `/home/workspace` — Scratch space for temp files and intermediate work. Not visible externally.
- `/mnt/user-data/outputs` — Copy selected **original files** (not preprocessed folders) from `/mnt/user-data/uploads` here as question attachments. Don't include other files.

[Your Core Tasks]
1. Thoroughly understand the content of the provided files.
2. Based on the workflow and file content, design a question that requires producing a real deliverable.
4. Generate verifiable rubric scoring criteria.

[Question Design Principles]
- You must analyze the files in detail before designing questions. Do not skip file analysis and jump directly to question design.
- Questions should reflect real-world work requirements, require multi-step reasoning and operations, and be closely related to the attachment content.
- Deliverable types: pdf, pptx, xlsx. Choose the deliverable type that best fits the task nature.
- Not all provided files need to be included, you can choose the most suitable files for the question.
- You can adjust and enrich the workflow appropriately based on the file content.
- **Keep the question description concise**: do NOT embed all details from the attachments into the question text. The question should set the task goal and context; solvers are expected to open the attachments and infer the specifics themselves.
- Questions should ideally test the model's ability to extract information from complex file structures (e.g., nested tables, multi-sheet workbooks, structured spreadsheets, or hierarchically organized documents).

[Rubric Rules]
1. **Source-backed and verifiable**: Each criterion must clearly determine "satisfied/not satisfied"; the basis must come from the question description or attachment content. Do not fabricate requirements or values without a source.
2. **Content category** (category=content): Assesses information retrieval, multi-step reasoning, and specific operations required to complete the task, e.g., extracting key data from attachments, calculating or summarizing according to specified logic, executing specific analysis steps.
3. **Form category** (category=form): Assesses the structural and formatting requirements of the deliverable, e.g., file format, section structure, number of charts, naming conventions.

[Return Format]
After completing file analysis, first copy the selected attachments from /mnt/user-data/uploads to /mnt/user-data/outputs, then return the following JSON:
{{
    "question": {{
        "title": "Question title",
        "description": "Detailed question description (background context, task requirements, delivery requirements)",
        "deliverable": {{
            "type": "pdf|pptx|xlsx",
            "filename": "Suggested filename"
        }},
        "attachments": [
            {{
                "filename": "Attachment filename (already copied to outputs directory)",
                "reason": "Reason for selecting this attachment"
            }}
        ]
    }},
    "rubric": [
        {{
            "criterion": "Specific verifiable scoring description",
            "category": "content|form",
            "points": 2,
            "source": "Source basis for the rubric and the approach needed to fulfill it"
        }}
    ]
}}

You have at most {max_steps} steps. Return the result when your task is complete."""


def build_question_user_prompt(
    scenario: Dict[str, Any],
    workflow: Dict[str, Any],
    matched_files: List[Dict[str, Any]],
    title: str,
    search_info: List[Dict[str, Any]] = None,
    only_raw_files: bool = False,
) -> str:
    """Build the user prompt for the Question Design Agent.

    Args:
        scenario: Scenario description
        workflow: Workflow description
        matched_files: List of files matching the requirements
        title: Job title
        search_info: Search metadata list (containing title, snippet, etc.)
        only_raw_files: True = only source files in sandbox, no preprocessed content; False = both source files and preprocessed content uploaded
    """
    scenario_text = f"""Scenario title: {scenario.get('title', '')}"""

    workflow_steps = workflow if isinstance(workflow, list) else []
    workflow_text = "\n".join([
        f"{i+1}. {step.get('description', '')}"
        for i, step in enumerate(workflow_steps)
    ])

    # Retrieved files (sandbox path + relevance + relevance type)
    relevance_type_map = {
        "highly_relevant": "Highly Relevant",
        "partially_relevant": "Partially Relevant",
    }

    # Build file path -> search info mapping
    search_info = search_info or []
    search_info_map = {}
    for s in search_info:
        fp = s.get('file_path', '')
        title = s.get('title', '')
        snippet = s.get('snippet', '')
        info_str = f"{title} — {snippet}" if title and snippet else title or snippet or ''
        if fp:
            search_info_map[fp] = info_str

    def _file_entry(f: Dict[str, Any]) -> str:
        sandbox_path = f.get('sandbox_path', '')
        relevance_type = f.get('relevance_type', '')
        relevance_type_label = relevance_type_map.get(relevance_type, relevance_type)
        relevance = f.get('relevance', 'N/A')
        # Preprocessed folder: same level and same name as file (without extension)
        if sandbox_path:
            from pathlib import PurePosixPath
            p = PurePosixPath(sandbox_path)
            preprocessed_dir = str(p.parent / p.stem)
        else:
            preprocessed_dir = ''
        lines = [
            f"- Path: {sandbox_path}",
            f"  Relevance: {relevance}",
        ]
        # Append search info (title + snippet combined) next to the file entry
        info = search_info_map.get(sandbox_path, '')
        if info:
            lines.append(f"  Info: {info}")
        if preprocessed_dir and not only_raw_files and relevance_type == "highly_relevant":
            lines.append(f"  Preprocessed folder: {preprocessed_dir} (for understanding only — contains content.md text and extracted images; DO NOT use as a question attachment)")
        return "\n".join(lines)

    files_text = "\n".join([_file_entry(f) for f in matched_files]) if matched_files else "No matching files available"

    return f"""Please design a productivity task question and generate rubric scoring criteria based on the following workflow scenario and retrieved files.

[Job Title] {title}

[Scenario Description]
{scenario_text}

[Workflow Description]
{workflow_text}

[Relevant Files] (Uploaded to /mnt/user-data/uploads, annotated with relevance type)
Note: Preprocessed folders are for reference only to help you understand file content. Only original files may be used as question attachments.
{files_text}

Please start analyzing the files and designing the question, then return the result in JSON format."""
