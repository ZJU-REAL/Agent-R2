"""Shared helpers for SFT/RL data preparation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

DEFAULT_THRESHOLD = 6000
DEFAULT_MAX_PROMPT_TOKENS = 60000


class SampleSplit(str, Enum):
    SFT = "sft"  # sample_idx < threshold
    RL = "rl"  # sample_idx >= threshold


@dataclass
class CollectStats:
    skipped_tokens: int = 0


def load_solver_prefix() -> str:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return cfg.get("agents", {}).get("solver_agent", {}).get(
            "family", cfg.get("defaults", {}).get("family", "")
        )
    except Exception:
        return ""


SOLVER_PREFIX = load_solver_prefix()


def add_prepare_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help=(
            f"Sample index cutoff (default {DEFAULT_THRESHOLD}). "
            "SFT: index < threshold; RL: index >= threshold."
        ),
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=DEFAULT_MAX_PROMPT_TOKENS,
        help=(
            f"Skip samples whose last assistant prompt_tokens exceeds this value "
            f"(default {DEFAULT_MAX_PROMPT_TOKENS}, same metric as statistics.py)."
        ),
    )


def parse_sample_idx(sample_dir: Path) -> Optional[int]:
    try:
        return int(sample_dir.name.split("_", 1)[1])
    except (ValueError, IndexError):
        return None


def passes_index_threshold(idx: int, threshold: int, split: SampleSplit) -> bool:
    if split == SampleSplit.SFT:
        return idx < threshold
    return idx >= threshold


def get_solver_file_paths(sample_dir: Path, prefix: str = SOLVER_PREFIX) -> Tuple[Path, Path, Path]:
    question_file = sample_dir / "question_result.json"
    result_name = f"{prefix}_solver_result.json" if prefix else "solver_result.json"
    conv_name = f"{prefix}_solver_agent.json" if prefix else "solver_agent.json"
    solver_result_file = sample_dir / result_name
    conv_file = sample_dir / "conversation" / conv_name
    return question_file, solver_result_file, conv_file


def get_last_solver_prompt_tokens(messages: List[Dict[str, Any]]) -> Optional[int]:
    """Last assistant message's prompt_tokens + completion_tokens."""
    last_tokens: Optional[int] = None
    for item in messages:
        if item.get("role") == "assistant":
            usage = item.get("usage", {})
            if usage and "prompt_tokens" in usage:
                last_tokens = usage["prompt_tokens"] + usage.get("completion_tokens", 0)
    return last_tokens


def exceeds_prompt_token_limit(messages: List[Dict[str, Any]], max_prompt_tokens: int) -> bool:
    tokens = get_last_solver_prompt_tokens(messages)
    if tokens is None:
        return False
    return tokens > max_prompt_tokens


def _load_conversation_messages(conv_file: Path, sample_name: str) -> Optional[List[Dict[str, Any]]]:
    try:
        return json.loads(conv_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  Skipping {sample_name}: failed to read solver_agent.json: {e}")
        return None


def _passes_sample_filters(
    sample_dir: Path,
    threshold: int,
    max_prompt_tokens: int,
    split: SampleSplit,
    stats: CollectStats,
) -> Optional[List[Dict[str, Any]]]:
    idx = parse_sample_idx(sample_dir)
    if idx is None or not passes_index_threshold(idx, threshold, split):
        return None

    question_file, solver_result_file, conv_file = get_solver_file_paths(sample_dir)
    if not question_file.exists() or not conv_file.exists() or not solver_result_file.exists():
        return None

    try:
        q_data = json.loads(question_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  Skipping {sample_dir.name}: failed to read question_result.json: {e}")
        return None

    if q_data.get("status", "") != "completed":
        return None

    messages = _load_conversation_messages(conv_file, sample_dir.name)
    if messages is None:
        return None

    if exceeds_prompt_token_limit(messages, max_prompt_tokens):
        stats.skipped_tokens += 1
        return None

    return messages


def _print_token_skip_summary(max_prompt_tokens: int, stats: CollectStats) -> None:
    if stats.skipped_tokens:
        print(
            f"  Skipped {stats.skipped_tokens} samples with "
            f"last assistant prompt_tokens > {max_prompt_tokens}"
        )


def collect_sft_samples(
    input_dir: Path,
    threshold: int = DEFAULT_THRESHOLD,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
) -> List[Tuple[Path, List[Dict[str, Any]]]]:
    samples: List[Tuple[Path, List[Dict[str, Any]]]] = []
    stats = CollectStats()

    for sample_dir in sorted(input_dir.glob("sample_*")):
        messages = _passes_sample_filters(
            sample_dir, threshold, max_prompt_tokens, SampleSplit.SFT, stats
        )
        if messages is not None:
            samples.append((sample_dir, messages))

    _print_token_skip_summary(max_prompt_tokens, stats)
    return samples


def collect_rl_sample_dirs(
    input_dir: Path,
    threshold: int = DEFAULT_THRESHOLD,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
) -> List[Path]:
    sample_dirs: List[Path] = []
    stats = CollectStats()

    for sample_dir in sorted(input_dir.glob("sample_*")):
        messages = _passes_sample_filters(
            sample_dir, threshold, max_prompt_tokens, SampleSplit.RL, stats
        )
        if messages is not None:
            sample_dirs.append(sample_dir)

    _print_token_skip_summary(max_prompt_tokens, stats)
    return sample_dirs
