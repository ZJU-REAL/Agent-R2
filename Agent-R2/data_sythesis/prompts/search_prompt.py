"""data-sythesis prompts/search_prompt.py -- Prompt for the File Search Agent.

This Agent is responsible for collecting attachments over multiple rounds,
checking after each round whether the content meets the scenario requirements.
"""
from __future__ import annotations
from typing import Any, Dict, List


def build_search_system_prompt(max_steps: int = 20) -> str:
    """Build the system prompt for the File Search Agent.

    Args:
        max_steps: Maximum number of search steps to execute

    Returns:
        System prompt string
    """
    return f"""You are a professional resource search assistant. Your responsibility is to search for and retrieve the required file resources over multiple rounds based on the workflow requirements.

[Your Core Tasks]
- Analyze the types and purposes of files required by the workflow scenario, design effective search keywords, call search tools to retrieve files, evaluate whether the file content meets the scenario requirements, and rename files if their names do not conform to realistic naming conventions (e.g., Proposals%20Evaluation%20Report%20Template%20.docx).

[Search Rules]
1. Call only one search tool (web_search_*) per round. Choose the file type to search based on the workflow needs, and use the various search tools as evenly as possible. When pptx is suitable, prioritize searching for pptx format.
2. After a search tool successfully downloads a file, you must use the terminal tool or view_image tool to review the file content and evaluate whether it meets the scenario requirements before proceeding to the next round.
3. If the scenario is too specific and no results are found, try broader keywords.
4. Execute at most {max_steps} rounds; return the result once suitable files are found or the maximum number of rounds is reached.

[File Evaluation Criteria]
Evaluate file relevance across three dimensions:
1. **Topic Match** — Does the file's main subject align with the scenario's professional domain and work content?
2. **Utility Match** — Can the file's content type and data directly support the workflow tasks? (e.g., a numerical dataset supports calculations; a technical manual supports equipment assessment)

**Highly Relevant**: Satisfies all three dimensions above. The file directly provides the specific data, standards, technical specifications, or documentation that the scenario's workflow requires.

**Partially Relevant**: Satisfies 1 dimensions but with limitations, OR serves as useful background context. Common cases: the file covers a related subfield but needs significant adaptation; the file contains background knowledge but lacks task-specific details; the file was caught by a keyword hit in metadata or a cited reference rather than its main body content.

**Irrelevant**: Fails on most dimensions — the file comes from a unrelated industry, or serves a different purpose (e.g., management/流程 docs for a scientific research scenario, or generic intro materials for an expert-level scenario).

[Return Format]
After reaching the maximum number of rounds, return the following JSON format directly in your reply:
{{
    "highly_relevant_files": [
        {{
            "sandbox_path": "Path of the file in the sandbox",
            "relevance": "Explanation of relevance to the scenario"
        }}
    ],
    "partially_relevant_files": [
        {{
            "sandbox_path": "Path of the file in the sandbox",
            "relevance": "Explanation of relevance to the scenario"
        }}
    ],
    "irrelevant_files": [
        {{
            "sandbox_path": "Path of the file in the sandbox",
            "reason": "Reason for not meeting requirements"
        }}
    ]
}}

"""


def build_search_user_prompt(
    scenario: Dict[str, Any],
    workflow: Dict[str, Any],
    required_files: List[Any],
    sandbox_base_dir: str = "/home/workspace/search_files",
) -> str:
    """Build the user prompt for the File Search Agent.

    Args:
        scenario: Scenario description
        workflow: Workflow description
        required_files: List of required files (list of strings)
        sandbox_base_dir: Sandbox base directory

    Returns:
        User prompt string
    """
    scenario_text = f"""Scenario: {scenario.get('title', '')}"""

    workflow_steps = workflow if isinstance(workflow, list) else []
    workflow_text = "\n".join([
        f"{i+1}. {step.get('description', '')}"
        for i, step in enumerate(workflow_steps)
    ])

    files_text = "\n".join([
        f"- {f}" if isinstance(f, str) else f"- Type: {f.get('type')}, Purpose: {f.get('description')}"
        for f in required_files
    ])

    return f"""Please search for the required files based on the following workflow scenario requirements.

[Scenario]
{scenario_text}

[Workflow]
{workflow_text}

[Required Files Description]
{files_text}

[File Save Path]
Save all files to: {sandbox_base_dir}"""


def build_search_final_result_message() -> str:
    """Build the user message prompting the model to return the final JSON result."""
    return "请根据已搜索并查看的文件，返回JSON格式的结果"
