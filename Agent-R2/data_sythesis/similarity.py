import json
import asyncio
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
import httpx
import openpyxl

ENDPOINT = "http://localhost:8000/v1/embeddings"
MODEL = "Qwen3-Embedding-4B"
BASE = Path(__file__).parent
K_CLUSTERS = 50  # for cluster inertia
MODE = "global"  # "per_discipline" | "global" | "both"

TEXT_FILES = {
    "workflow": BASE / "workflow_texts.json",
    "task": BASE / "task_texts.json",
    "question": BASE / "question_texts.json",
}


async def embed_one(text: str, client: httpx.AsyncClient) -> list[float]:
    resp = await client.post(ENDPOINT, json={"model": MODEL, "input": text})
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


async def embed_all(texts: list[str], client: httpx.AsyncClient) -> list[list[float]]:
    return await asyncio.gather(*[embed_one(t, client) for t in texts])


def normalize(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(norms == 0, 1, norms)


def compute_metrics(embeddings: np.ndarray) -> dict:
    """Compute all 5 diversity metrics for a set of embeddings."""
    n, d = embeddings.shape
    normed = normalize(embeddings)

    # Mean Cosine Distance: average 1 - cos(ei, ej) over all pairs
    sim_mat = normed @ normed.T
    upper = sim_mat[np.triu_indices(n, k=1)]
    mean_cosine_dist = float(np.mean(1 - upper))

    # Mean L2 Distance
    # use pairwise via broadcasting, chunked to save memory
    diff = embeddings[:, None, :] - embeddings[None, :, :]  # n x n x d
    l2_mat = np.sqrt((diff ** 2).sum(axis=-1))
    upper_l2 = l2_mat[np.triu_indices(n, k=1)]
    mean_l2_dist = float(np.mean(upper_l2))

    # 1-NN Distance: for each point, cosine distance to nearest neighbor
    np.fill_diagonal(sim_mat, -2)  # exclude self
    nn_cos_sim = sim_mat.max(axis=1)
    mean_1nn_dist = float(np.mean(1 - nn_cos_sim))

    # Cluster Inertia (K-Means)
    k = min(K_CLUSTERS, n - 1)
    km = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42)
    km.fit(embeddings)
    inertia = float(km.inertia_)

    # Radius: geometric mean of per-dimension std
    stds = embeddings.std(axis=0)
    radius = float(np.exp(np.mean(np.log(stds + 1e-12))))

    return {
        "mean_cosine_distance": mean_cosine_dist,
        "mean_l2_distance": mean_l2_dist,
        "mean_1nn_distance": mean_1nn_dist,
        "cluster_inertia": inertia,
        "radius": radius,
    }


async def compute_one(group_name: str, texts_file: Path, client: httpx.AsyncClient) -> dict:
    with open(texts_file) as f:
        by_discipline = json.load(f)

    discipline_results = {}
    all_embeddings = []

    if MODE in ("per_discipline", "both"):
        for discipline, items in by_discipline.items():
            if len(items) < 2:
                continue
            embs = np.array(await embed_all([item["text"] for item in items], client))
            all_embeddings.append(embs)
            metrics = compute_metrics(embs)
            discipline_results[discipline] = {"count": len(items), **metrics}
            print(f"[{group_name}] {discipline}: n={len(items)}, cosine_dist={metrics['mean_cosine_distance']:.4f}")
    else:
        for discipline, items in by_discipline.items():
            if len(items) < 2:
                continue
            embs = np.array(await embed_all([item["text"] for item in items], client))
            all_embeddings.append(embs)

    # Global metrics across all disciplines
    global_metrics = {}
    if MODE in ("global", "both"):
        global_embs = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, 1))
        global_metrics = compute_metrics(global_embs) if len(global_embs) >= 2 else {}
        print(f"[{group_name}] GLOBAL n={len(global_embs)}, cosine_dist={global_metrics.get('mean_cosine_distance', 0):.4f}\n")

    # Within-discipline averages
    within_avg = {}
    if discipline_results:
        for key in ["mean_cosine_distance", "mean_l2_distance", "mean_1nn_distance", "cluster_inertia", "radius"]:
            within_avg[key] = float(np.mean([v[key] for v in discipline_results.values()]))

    return {
        "disciplines": discipline_results,
        "within_discipline_avg": within_avg,
        "global": global_metrics,
    }


async def compute():
    all_results = {}
    async with httpx.AsyncClient(timeout=120) as client:
        for group_name, texts_file in TEXT_FILES.items():
            print(f"=== {group_name} ===")
            all_results[group_name] = await compute_one(group_name, texts_file, client)

    out_file = BASE / "similarity_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Write xlsx
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    groups = list(all_results.keys())
    metrics = ["mean_cosine_distance", "mean_l2_distance", "mean_1nn_distance", "cluster_inertia", "radius"]

    # Sheet 1: per-discipline
    ws1 = wb.create_sheet("per_discipline")
    header = ["Discipline", "Count"]
    for g in groups:
        for m in metrics:
            header.append(f"{g}_{m}")
    ws1.append(header)

    all_disciplines = sorted({d for g in groups for d in all_results[g]["disciplines"]})
    for disc in all_disciplines:
        row = [disc]
        count = next((all_results[g]["disciplines"][disc]["count"]
                      for g in groups if disc in all_results[g]["disciplines"]), "")
        row.append(count)
        for g in groups:
            for m in metrics:
                row.append(all_results[g]["disciplines"].get(disc, {}).get(m, ""))
        ws1.append(row)

    # Sheet 2: summary (within-discipline avg + global)
    ws2 = wb.create_sheet("summary")
    ws2.append([""] + groups)
    for m in metrics:
        row_within = [f"within_avg_{m}"]
        row_global = [f"global_{m}"]
        for g in groups:
            row_within.append(all_results[g].get("within_discipline_avg", {}).get(m, ""))
            row_global.append(all_results[g].get("global", {}).get(m, ""))
        ws2.append(row_within)
        ws2.append(row_global)

    xlsx_file = BASE / "similarity_results.xlsx"
    wb.save(xlsx_file)
    print(f"Saved to {out_file} and {xlsx_file}")


if __name__ == "__main__":
    asyncio.run(compute())
