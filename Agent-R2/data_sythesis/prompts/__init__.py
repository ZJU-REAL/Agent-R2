"""data-sythesis prompts package."""
from __future__ import annotations

from .scenario_prompt import build_scenario_system_prompt, build_scenario_user_prompt
from .search_prompt import build_search_system_prompt, build_search_user_prompt
from .question_prompt import (
    build_question_system_prompt,
    build_question_user_prompt,
)
from .solver_prompt import (
    build_solver_system_prompt,
    build_solver_user_prompt,
)

__all__ = [
    "build_scenario_system_prompt",
    "build_scenario_user_prompt",
    "build_search_system_prompt",
    "build_search_user_prompt",
    "build_question_system_prompt",
    "build_question_user_prompt",
    "build_solver_system_prompt",
    "build_solver_user_prompt",
]
