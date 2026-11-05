"""Prepare SFT dataset (split): one trajectory -> multiple per-turn samples.

Uses samples with index < --threshold (before threshold).
Turns whose following tool_response contains an error are skipped.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prepare_utils import add_prepare_filter_args, collect_sft_samples
from tool_manager import get_solver_tools


def format_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
    parts = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        arguments_raw = func.get("arguments", "{}")
        try:
            arguments = (
                json.loads(arguments_raw)
                if isinstance(arguments_raw, str)
                else arguments_raw
            )
        except json.JSONDecodeError:
            arguments = {}
        call_obj = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
        parts.append(f"<tool_call>\n{call_obj}\n</tool_call>")
    return "\n".join(parts)


def merge_assistant_turn(reasoning: str, content: str, tool_calls: Optional[List]) -> str:
    parts = []
    if reasoning:
        parts.append(f"<think>\n{reasoning.strip()}\n</think>")
    if content:
        parts.append(content.strip())
    if tool_calls:
        parts.append(format_tool_calls(tool_calls))
    return "\n".join(parts)


def process_tool_result(content: Any) -> str:
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            return json.dumps(parsed, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return content
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def build_swift_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    swift_messages: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            swift_messages.append({"role": "system", "content": msg.get("content", "")})
        elif role == "user":
            swift_messages.append({"role": "user", "content": msg.get("content", "")})
        elif role == "assistant":
            reasoning = msg.get("reasoning_content", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            merged = merge_assistant_turn(reasoning, content, tool_calls)
            if merged:
                swift_messages.append({"role": "assistant", "content": merged, "loss": True})
        elif role == "tool":
            resp_content = msg.get("content", "")
            processed = process_tool_result(resp_content)
            swift_messages.append({"role": "tool_response", "content": processed})
    return swift_messages


def find_turn_boundaries(swift_messages: List[Dict[str, Any]]) -> List[int]:
    return [i for i, msg in enumerate(swift_messages) if msg.get("role") == "assistant"]


def _is_tool_error(content: str) -> bool:
    if content.startswith("{"):
        try:
            return "error" in json.loads(content)
        except (json.JSONDecodeError, ValueError):
            pass
    return False


def expand_trajectory(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    swift_messages = build_swift_messages(messages)
    boundaries = find_turn_boundaries(swift_messages)
    if not boundaries:
        return [], 0

    tools_str = json.dumps(get_solver_tools(), ensure_ascii=False)
    samples: List[Dict[str, Any]] = []
    skipped_error = 0

    for k, turn_end in enumerate(boundaries):
        has_error = False
        next_idx = turn_end + 1
        while next_idx < len(swift_messages) and swift_messages[next_idx].get("role") == "tool_response":
            if _is_tool_error(swift_messages[next_idx].get("content", "")):
                has_error = True
                break
            next_idx += 1

        if has_error:
            skipped_error += 1
            continue

        slice_msgs = swift_messages[: turn_end + 1]
        prev_turn_end = boundaries[k - 1] if k > 0 else -1
        current_turn_start = prev_turn_end + 1

        sample_messages = []
        for idx, msg in enumerate(slice_msgs):
            if idx < current_turn_start:
                m = {key: v for key, v in msg.items() if key != "loss"}
                if m.get("role") == "assistant" and "<think>" in m.get("content", ""):
                    m = dict(m)
                    m["content"] = re.sub(
                        r"<think>.*?</think>\n?", "", m["content"], flags=re.DOTALL
                    ).lstrip()
            else:
                m = dict(msg)
            sample_messages.append(m)

        samples.append({"tools": tools_str, "messages": sample_messages})

    return samples, skipped_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT data (per-turn split, index < threshold)")
    add_prepare_filter_args(parser)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    input_dir = project_root / "output"
    output_dir = project_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_samples = collect_sft_samples(input_dir, args.threshold, args.max_prompt_tokens)
    print(f"Valid SFT samples (index < {args.threshold}): {len(sft_samples)}")
    if not sft_samples:
        print("No valid samples found!")
        return

    sft_output = output_dir / "sft.jsonl"
    total_rows = 0
    errors = 0
    total_error_skipped = 0

    with open(sft_output, "w", encoding="utf-8") as out_f:
        for sample_dir, messages in sft_samples:
            try:
                samples, skipped = expand_trajectory(messages)
                total_error_skipped += skipped
                if not samples:
                    continue
                for sample in samples:
                    out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total_rows += len(samples)
            except Exception as e:
                print(f"  SFT error {sample_dir.name}: {e}")
                errors += 1

    print(
        f"\nSFT: {len(sft_samples)} trajectories -> {total_rows} training samples "
        f"({errors} errors, {total_error_skipped} turns skipped by error filter) -> {sft_output}"
    )


if __name__ == "__main__":
    main()
