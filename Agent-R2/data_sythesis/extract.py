"""提取所有样本的workflow/task/question文本，按discipline分组写入三个json文件"""
import json
from pathlib import Path
from collections import defaultdict

output_dir = Path(__file__).parent / "output"
base = Path(__file__).parent

workflows = defaultdict(list)
tasks = defaultdict(list)
questions = defaultdict(list)

for sample_dir in sorted(output_dir.iterdir()):
    scenario_file = sample_dir / "scenario_result.json"
    if not scenario_file.exists():
        continue
    with open(scenario_file) as f:
        s = json.load(f)
    discipline = s.get("discipline")
    if not discipline:
        continue
    name = sample_dir.name

    # workflow
    steps = s.get("scenario", {}).get("workflow", [])
    workflow_text = " ".join(step.get("description", "") for step in steps)
    workflows[discipline].append({"sample": name, "text": workflow_text})

    # task
    task_list = s.get("task", [])
    task_text = " ".join(task_list) if isinstance(task_list, list) else str(task_list)
    tasks[discipline].append({"sample": name, "text": task_text})

    # question
    question_file = sample_dir / "question_result.json"
    if question_file.exists():
        with open(question_file) as f:
            q = json.load(f)
        desc = (q.get("question") or {}).get("description", "")
        questions[discipline].append({"sample": name, "text": desc})

for fname, data in [
    ("workflow_texts.json", workflows),
    ("task_texts.json", tasks),
    ("question_texts.json", questions),
]:
    out = {d: items for d, items in data.items() if len(items) >= 2}
    with open(base / fname, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    total = sum(len(v) for v in out.values())
    print(f"{fname}: {total} samples, {len(out)} disciplines")
