"""从已有的 eval_result.json 生成 summary.json"""
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from officeqa.utils import save_summary

_RESULTS_ROOT = Path(__file__).resolve().parent / "results"


def main(output_dir: str) -> None:
    d = Path(output_dir)
    results = []
    for task_dir in sorted(d.iterdir()):
        eval_file = task_dir / "eval_result.json"
        if eval_file.exists():
            with open(eval_file, encoding="utf-8") as f:
                results.append(json.load(f))

    path = save_summary(results, output_dir)
    acc = sum(r.get("accuracy", 0) for r in results) / len(results) if results else 0
    print(f"num_tasks={len(results)}, accuracy={acc:.4f}, saved to {path}")


if __name__ == "__main__":
    main(str(_RESULTS_ROOT / "qwen3-235b-a22b-thinking-2507"))
