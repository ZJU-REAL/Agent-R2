import json
from pathlib import Path

OFFICE_EXTS = {'.pptx', '.pdf', '.docx', '.xlsx','.doc','.csv','.ppt'}
GDPVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = GDPVAL_DIR / 'results'
DATA_FILE = GDPVAL_DIR / 'data' / 'train-00000-of-00001.json'
SPLIT_JSON = RESULTS_DIR / 'office_split.json'

TARGET_MODELS = [
    'claude-opus-4-5-20251101-kiro',
    'gemini-2.5-pro',
    'gpt-4.1',
    'glm-4.7',
    'deepseek-v3.1',
    'Qwen3-4B-Thinking-2507',
    'qwen3-30b-a3b-thinking-2507',
    'Qwen3-4B-Thinking-2507-rl80_3',
    'qwen3-235b-a22b-thinking-2507',
    'qwen3-vl-30b-a3b-thinking',
    'qwen3-vl-235b-a22b-thinking',
]


def classify_samples_from_data() -> dict:
    with open(DATA_FILE) as f:
        tasks = json.load(f)
    office, non_office = [], []
    for task in tasks:
        files = task.get('deliverable_files', [])
        if files and all(Path(f).suffix.lower() in OFFICE_EXTS for f in files):
            office.append(task['task_id'])
        else:
            non_office.append(task['task_id'])
    return {'office': office, 'non_office': non_office}


def load_scores(model_dir: Path, sample_ids: list) -> list:
    scores = []
    for sid in sample_ids:
        result_file = model_dir / sid / 'eval_result.json'
        if not result_file.exists():
            scores.append(0.0)  # 与 summary.json 保持一致，缺失记 0 分
            continue
        with open(result_file) as f:
            data = json.load(f)
        evals = data.get('evaluations', [])
        if not evals:
            sc, mx = data.get('score', 0), data.get('max_score', 0)
            scores.append(sc / mx if mx > 0 else 0.0)
            continue
        s_fixed, m_fixed = 0, 0
        for e in evals:
            ms, sc = e.get('max_score', 0), e.get('score', 0)
            if ms > 0:
                s_fixed += sc; m_fixed += ms
            elif ms < 0:
                s_fixed += sc - ms; m_fixed += -ms
        scores.append(s_fixed / m_fixed if m_fixed > 0 else 0.0)
    return scores


def compute_metrics(scores: list) -> dict:
    if not scores:
        return {'count': 0, 'avg_score': 0, 'avg_normalized': 0}
    return {
        'count': len(scores),
        'avg_normalized': sum(scores) / len(scores),
    }


# Step 1: Build split from data file
print(f'Classifying samples from data file...')
split = classify_samples_from_data()
print(f"Office: {len(split['office'])}, Non-office: {len(split['non_office'])}")

with open(SPLIT_JSON, 'w') as f:
    json.dump(split, f, indent=2)
print(f'Saved split to {SPLIT_JSON}')

# Step 2: Load split and compute metrics per model
with open(SPLIT_JSON) as f:
    split = json.load(f)

office_ids = set(split['office'])
non_office_ids = set(split['non_office'])

print('\n=== Metrics by model ===')
header = f"{'Model':<45} {'Group':<12} {'Count':>6} {'Accuracy':>10}"
print(header)
print('-' * len(header))

for model in TARGET_MODELS:
    model_dir = RESULTS_DIR / model
    if not model_dir.exists():
        print(f'{model}: directory not found, skipping')
        continue

    all_samples = {s.name for s in model_dir.iterdir() if s.is_dir()}
    office_in_model = sorted(all_samples & office_ids)
    non_office_in_model = sorted(all_samples & non_office_ids)

    all_in_model = sorted(all_samples & (office_ids | non_office_ids))
    for group_name, ids in [('office', office_in_model), ('non_office', non_office_in_model), ('all', all_in_model)]:
        scores = load_scores(model_dir, ids)
        m = compute_metrics(scores)
        print(f'{model:<45} {group_name:<12} {m["count"]:>6} {m["avg_normalized"]:>10.4f}')
