"""Prepare SFT dataset (non-split): one trajectory -> one sample.

Uses samples with index < --threshold (before threshold).
When a tool_response contains an error, the preceding assistant turn gets loss=False
(instead of dropping the whole trajectory).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _is_tool_error(content: str) -> bool:
    if content.startswith("{"):
        try:
            return "error" in json.loads(content)
        except (json.JSONDecodeError, ValueError):
            pass
    return False


def build_swift_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert raw conversation to ms-swift format; set loss per assistant turn."""
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
                swift_messages.append({"role": "assistant", "content": merged})
        elif role == "tool":
            resp_content = msg.get("content", "")
            processed = process_tool_result(resp_content)
            swift_messages.append({"role": "tool_response", "content": processed})

    for i, msg in enumerate(swift_messages):
        if msg.get("role") != "assistant":
            continue
        has_error = False
        j = i + 1
        while j < len(swift_messages) and swift_messages[j].get("role") == "tool_response":
            if _is_tool_error(swift_messages[j].get("content", "")):
                has_error = True
                break
            j += 1
        msg["loss"] = not has_error

    return swift_messages


def build_trajectory(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    swift_messages = build_swift_messages(messages)
    if not any(msg.get("role") == "assistant" for msg in swift_messages):
        return None
    tools_str = json.dumps(get_solver_tools(), ensure_ascii=False)
    return {"tools": tools_str, "messages": swift_messages}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT data (non-split, index < threshold)")
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
    loss_false_turns = 0
    errors = 0
    skipped_no_assistant = 0

    with open(sft_output, "w", encoding="utf-8") as out_f:
        for sample_dir, messages in sft_samples:
            try:
                sample = build_trajectory(messages)
                if sample is None:
                    skipped_no_assistant += 1
                    continue
                for msg in sample["messages"]:
                    if msg.get("role") == "assistant" and msg.get("loss") is False:
                        loss_false_turns += 1
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total_rows += 1
            except Exception as e:
                print(f"  SFT error {sample_dir.name}: {e}")
                errors += 1

    print(
        f"\nSFT: {len(sft_samples)} trajectories -> {total_rows} samples "
        f"({errors} errors, {skipped_no_assistant} skipped: no assistant turns, "
        f"{loss_false_turns} assistant turns with loss=False) -> {sft_output}"
    )


if __name__ == "__main__":
    main()
