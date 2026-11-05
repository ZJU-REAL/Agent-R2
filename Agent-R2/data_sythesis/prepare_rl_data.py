"""Prepare RL dataset from sample_* directories.

Uses samples with index >= --threshold (after threshold).
Outputs batched zip files under rl_data/.
"""

import argparse
import zipfile
from pathlib import Path
from typing import List

from prepare_utils import add_prepare_filter_args, collect_rl_sample_dirs, collect_sft_samples

BATCH_SIZE = 2000


def write_sample_to_zip(zf: zipfile.ZipFile, sample_dir: Path) -> None:
    qfile = sample_dir / "question_result.json"
    if qfile.exists():
        zf.write(qfile, arcname=f"{sample_dir.name}/question_result.json")
    qtmp = sample_dir / "question_tmp"
    if qtmp.exists():
        for fpath in qtmp.rglob("*"):
            if fpath.is_file():
                rel = fpath.relative_to(qtmp)
                zf.write(fpath, arcname=f"{sample_dir.name}/question_tmp/{rel}")


def collect_token_filtered_sft_dirs(
    input_dir: Path,
    threshold: int,
    max_prompt_tokens: int,
) -> List[Path]:
    """Collect SFT samples (index < threshold) that were filtered out by token limit.

    These are valid SFT candidates but exceeded max_prompt_tokens.
    """
    from prepare_utils import (
        SampleSplit,
        _passes_sample_filters,
        get_solver_file_paths,
        parse_sample_idx,
        passes_index_threshold,
    )
    import json

    dirs: List[Path] = []
    for sample_dir in sorted(input_dir.glob("sample_*")):
        idx = parse_sample_idx(sample_dir)
        if idx is None or not passes_index_threshold(idx, threshold, SampleSplit.SFT):
            continue

        question_file, solver_result_file, conv_file = get_solver_file_paths(sample_dir)
        if not question_file.exists() or not conv_file.exists() or not solver_result_file.exists():
            continue

        try:
            q_data = json.loads(question_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        if q_data.get("status", "") != "completed":
            continue

        from prepare_utils import exceeds_prompt_token_limit, _load_conversation_messages

        messages = _load_conversation_messages(conv_file, sample_dir.name)
        if messages is None:
            continue

        if exceeds_prompt_token_limit(messages, max_prompt_tokens):
            dirs.append(sample_dir)

    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare RL data (index >= threshold, plus token-filtered SFT samples)"
    )
    add_prepare_filter_args(parser)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    input_dir = project_root / "output"
    output_dir = project_root / "rl_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = collect_rl_sample_dirs(input_dir, args.threshold, args.max_prompt_tokens)
    print(f"Valid RL samples (index >= {args.threshold}): {len(sample_dirs)}")

    # Also collect SFT samples (index < threshold) that were filtered out by token limit
    token_filtered_dirs = collect_token_filtered_sft_dirs(
        input_dir, args.threshold, args.max_prompt_tokens
    )
    if token_filtered_dirs:
        print(
            f"Token-filtered SFT samples (index < {args.threshold}, "
            f"prompt_tokens > {args.max_prompt_tokens}): {len(token_filtered_dirs)}"
        )
        sample_dirs.extend(token_filtered_dirs)

    total = len(sample_dirs)
    if total == 0:
        print("No valid samples found!")
        return

    for i in range(0, total, BATCH_SIZE):
        batch = sample_dirs[i : i + BATCH_SIZE]
        zip_idx = i // BATCH_SIZE + 1
        zip_path = output_dir / f"data{zip_idx}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for sample_dir in batch:
                write_sample_to_zip(zf, sample_dir)
        print(f"Wrote {len(batch)} samples to {zip_path}")

    print(f"\nDone. Output dir: {output_dir}")


if __name__ == "__main__":
    main()
