"""data-sythesis prompts/scenario_prompt.py -- Prompts for scenario generation."""
from __future__ import annotations
from typing import List


def build_scenario_system_prompt() -> str:
    """Build the system prompt for the Scenario Generation Agent."""
    return """You are a professional productivity task scenario designer. Your responsibility is to analyze web corpus content, determine whether it is suitable for designing workflow scenarios, and when suitable, mine potential workflow scenarios from it.

[Your Core Tasks]
1. Deeply analyze the input web corpus, understanding its core themes, domain background, and key information.
2. Determine whether the corpus can support one or more real, complex, file-based workflow scenarios as described in the task description.
3. If so, mine potential workflows and required files based on the corpus content and task description, conducting multi-step workflows based on one or more files.
4. You may freely combine, decompose, or derive one or more task scenarios without being limited to the preset task descriptions.


[Workflow Design Principles]
- Authenticity: Scenarios should reflect real-world work requirements that exist in practice.
- Generalizability: Workflow steps should use generic sub-goal descriptions, avoiding overly specific entity information that would be hard to find relevant files for (e.g., specific chapter numbers, codes, etc.).
- Complexity: Workflows should have sufficient complexity, requiring multi-step reasoning and operations to complete.
- Reasonableness: Required files should conform to actual work habits in the relevant domain.

[Output Format]
Please strictly follow the JSON format below.

When the corpus is **not suitable** for question design (workflow is false):
```json
{
  "summary": "Brief summary of the web corpus content, describing its core theme",
  "workflow": false
}
```

When the corpus is **suitable** for question design (workflow is true):
```json
{
  "summary": "Brief summary of the web corpus content, describing its core theme",
  "workflow": true,
  "task": ["Reference task content verbatim", "Reference task content verbatim"],
  "scenario": {
    "title": "Scenario title",
    "workflow": [
      {"step": 1, "description": "Step 1 description"},
      {"step": 2, "description": "Step 2 description"}
    ]
  },
  "required_files": [
    "Description of file type and purpose",
    "Description of file type and purpose"
  ],
  "reasoning": "Rationale for designing this scenario, explaining how the web corpus supports this workflow"
}
```
"""


def build_scenario_user_prompt(
    original_document: str,
    title: str,
    candidate_tasks: List[str]
) -> str:
    """Build the user prompt for the Scenario Generation Agent.

    Args:
        original_document: Original document content (from web corpus)
        title: Corresponding job title
        candidate_tasks: List of candidate tasks (can be freely combined, decomposed, or derived)

    Returns:
        User prompt string
    """
    tasks_text = "\n".join([f"{i+1}. {task}" for i, task in enumerate(candidate_tasks)])

    return f"""Based on the following information, first determine whether the web corpus is suitable for question design, then decide whether to design a workflow scenario.

[Job Title]
{title}

[Web Corpus Content]
{original_document}

[Reference Tasks] (Fill the task field with the verbatim content of the tasks actually used)
{tasks_text}

Please deeply analyze the web corpus content:
1. First determine whether the corpus is suitable for question design (workflow true/false).
2. If suitable, mine potential workflow scenarios and fill the task field with the verbatim reference task content actually used.
3. Each item in required_files should be a single string describing the file type and purpose.

Output the result in JSON format."""
