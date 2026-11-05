import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / 'results'
DATA_FILE = Path(__file__).parent / 'data' / 'train-00000-of-00001.json'
SPLIT_JSON = RESULTS_DIR / 'office_split.json'

OFFICE_EXTS = {'.pptx', '.pdf', '.docx', '.xlsx', '.doc', '.csv', '.ppt'}

SFT_MODEL = 'Qwen3-4B-Thinking-2507-claude-ckpt2000'
RL_MODEL = 'Qwen3-4B-Thinking-2507-rl80_3'


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


def get_split() -> dict:
    if SPLIT_JSON.exists():
        with open(SPLIT_JSON) as f:
            return json.load(f)
    split = classify_samples_from_data()
    with open(SPLIT_JSON, 'w') as f:
        json.dump(split, f, indent=2)
    return split


def is_completed(model_dir: Path, sample_id: str) -> bool:
    """True if the sample's output dir exists and contains at least one file."""
    output_dir = model_dir / sample_id / 'output'
    if not output_dir.is_dir():
        return False
    return any(f.is_file() for f in output_dir.iterdir())


def load_score(model_dir: Path, sample_id: str):
    """Return normalized score [0,1], or None if eval_result.json missing."""
    result_file = model_dir / sample_id / 'eval_result.json'
    if not result_file.exists():
        return None
    with open(result_file) as f:
        data = json.load(f)
    evals = data.get('evaluations', [])
    if not evals:
        sc, mx = data.get('score', 0), data.get('max_score', 0)
        return sc / mx if mx > 0 else 0.0
    s_fixed, m_fixed = 0, 0
    for e in evals:
        ms, sc = e.get('max_score', 0), e.get('score', 0)
        if ms > 0:
            s_fixed += sc
            m_fixed += ms
        elif ms < 0:
            s_fixed += sc - ms
            m_fixed += -ms
    return s_fixed / m_fixed if m_fixed > 0 else 0.0


def compare_win_tie_loss(sample_ids, sft_dir: Path, rl_dir: Path):
    """
    Only includes samples where BOTH models are completed (output dir has files).
    win  = RL score > SFT score
    tie  = RL score == SFT score
    loss = RL score < SFT score
    """
    win = tie = loss = skipped = 0
    for sid in sample_ids:
        if not is_completed(sft_dir, sid) or not is_completed(rl_dir, sid):
            skipped += 1
            continue
        sft_score = load_score(sft_dir, sid)
        rl_score = load_score(rl_dir, sid)
        if sft_score is None or rl_score is None:
            skipped += 1
            continue
        if rl_score > sft_score:
            win += 1
        elif rl_score == sft_score:
            tie += 1
        else:
            loss += 1
    total = win + tie + loss
    return win, tie, loss, total, skipped


def main():
    split = get_split()
    office_ids = set(split['office'])
    all_ids = office_ids | set(split['non_office'])

    sft_dir = RESULTS_DIR / SFT_MODEL
    rl_dir = RESULTS_DIR / RL_MODEL

    sft_samples = {s.name for s in sft_dir.iterdir() if s.is_dir()}
    rl_samples = {s.name for s in rl_dir.iterdir() if s.is_dir()}
    both_samples = sft_samples & rl_samples

    print(f'SFT model:  {SFT_MODEL}')
    print(f'RL model:   {RL_MODEL}')
    print(f'Both dirs have {len(both_samples)} common sample dirs\n')

    groups = [
        ('office', both_samples & office_ids),
        ('all',    both_samples & all_ids),
    ]

    header = f"{'Group':<10} {'Win(RL>SFT)':>12} {'Tie':>6} {'Loss(RL<SFT)':>13} {'Total':>7} {'Skipped':>8}"
    print(header)
    print('-' * len(header))
    for group_name, ids in groups:
        ids = sorted(ids)
        win, tie, loss, total, skipped = compare_win_tie_loss(ids, sft_dir, rl_dir)
        print(f'{group_name:<10} {win:>12} {tie:>6} {loss:>13} {total:>7} {skipped:>8}')
        if total > 0:
            print(f'           Win: {win/total*100:.1f}%  Tie: {tie/total*100:.1f}%  Loss: {loss/total*100:.1f}%')
        print()


if __name__ == '__main__':
    main()
