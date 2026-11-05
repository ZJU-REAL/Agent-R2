"""prompts package."""
from __future__ import annotations

from .solver_prompt import (
    build_solver_system_prompt,
    build_solver_user_prompt,
)

__all__ = [
    "build_solver_system_prompt",
    "build_solver_user_prompt",
]
