#!/usr/bin/env bash
# Merge FSDP checkpoints saved across multiple Ray nodes into a single HF model.
#
# Usage (run on the machine that will hold the merged checkpoints, usually node0):
#   bash merge_multinode_checkpoints.sh
#
# Or override variables:
#   NODE1_HOST=<other-node-ip> STEP=global_step_10 bash merge_multinode_checkpoints.sh
# Defaults: RAY_PKG_DIR=_ray_pkg_54306093ad31188b; NODE1_HOST empty = all shards already local.
#
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────────
# Ray working_dir checkpoint source (same relative path on both nodes)
RAY_CKPT_SUBPATH="checkpoints/real_rl/solver_grpo_qwen3_4b_thinking_2507"

# Resolve Ray session working_dir
RAY_SESSION_ROOT="${RAY_SESSION_ROOT:-/tmp/ray/session_latest/runtime_resources/working_dir_files}"
RAY_PKG_DIR="${RAY_PKG_DIR:-_ray_pkg_54306093ad31188b}"

# verl repo root (for python -m verl.model_merger)
VERL_ROOT="${VERL_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}"

# Target: permanent checkpoint location
TARGET_ROOT="${TARGET_ROOT:-$VERL_ROOT/checkpoints/real_rl/solver_grpo_qwen3_4b_thinking_2507}"

# Other node (hostname or IP). Script copies local shards first, then fills any missing ranks 0..15 from here.
NODE1_HOST="${NODE1_HOST:-}"

# FSDP world size = nnodes * n_gpus_per_node (2 * 8 = 16)
WORLD_SIZE="${WORLD_SIZE:-16}"

# Which global_step to merge; empty = merge all global_step_* under TARGET_ROOT
STEP="${STEP:-}"

# HF output suffix under each global_step_*
HF_SUFFIX="${HF_SUFFIX:-actor_hf}"

# Skip merge step (only gather shards)
GATHER_ONLY="${GATHER_ONLY:-0}"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

resolve_ray_src() {
    local host_label="$1"
    local base="$2"

    if [[ -n "${RAY_SRC_OVERRIDE:-}" ]]; then
        echo "$RAY_SRC_OVERRIDE"
        return
    fi

    if [[ ! -d "$base" ]]; then
        die "Ray working_dir not found on ${host_label}: $base"
    fi

    # Use RAY_PKG_DIR (default: _ray_pkg_54306093ad31188b)
    if [[ -n "$RAY_PKG_DIR" ]]; then
        local src="${base}/${RAY_PKG_DIR}/${RAY_CKPT_SUBPATH}"
        [[ -d "$src" ]] || die "Not found: $src"
        echo "$src"
        return
    fi

    # Auto-detect _ray_pkg_* directory containing our checkpoint path
    local src=""
    for pkg in "$base"/_ray_pkg_*; do
        [[ -d "$pkg/$RAY_CKPT_SUBPATH" ]] || continue
        src="$pkg/$RAY_CKPT_SUBPATH"
        break
    done

    if [[ -z "$src" ]]; then
        die "Cannot find ${RAY_CKPT_SUBPATH} under any _ray_pkg_* in ${base} on ${host_label}"
    fi
    echo "$src"
}

count_model_shards() {
    local actor_dir="$1"
    ls -1 "$actor_dir"/model_world_size_"${WORLD_SIZE}"_rank_*.pt 2>/dev/null | wc -l | tr -d ' '
}

list_missing_ranks() {
    local actor_dir="$1"
    local missing=()
    local r
    for r in $(seq 0 $((WORLD_SIZE - 1))); do
        [[ -f "$actor_dir/model_world_size_${WORLD_SIZE}_rank_${r}.pt" ]] || missing+=("$r")
    done
    echo "${missing[*]}"
}

remote_actor_dir() {
    local step_name="$1"
    ssh "$NODE1_HOST" "bash -s" <<EOF
set -euo pipefail
base="${RAY_SESSION_ROOT}"
RAY_PKG_DIR="${RAY_PKG_DIR}"
RAY_CKPT_SUBPATH="${RAY_CKPT_SUBPATH}"
RAY_SRC_OVERRIDE="${RAY_SRC_OVERRIDE:-}"
resolve() {
  if [[ -n "\$RAY_SRC_OVERRIDE" ]]; then echo "\$RAY_SRC_OVERRIDE"; return; fi
  if [[ -n "\$RAY_PKG_DIR" ]]; then echo "\${base}/\${RAY_PKG_DIR}/\${RAY_CKPT_SUBPATH}"; return; fi
  for pkg in "\$base"/_ray_pkg_*; do
    [[ -d "\$pkg/\$RAY_CKPT_SUBPATH" ]] && { echo "\$pkg/\$RAY_CKPT_SUBPATH"; return; }
  done
  exit 1
}
echo "\$(resolve)/${step_name}/actor"
EOF
}

fetch_shard_from_remote() {
    local remote_actor="$1"
    local actor_dst="$2"
    local r="$3"
    local prefix f

    for prefix in model optim extra_state; do
        f="${prefix}_world_size_${WORLD_SIZE}_rank_${r}.pt"
        if ssh "$NODE1_HOST" "test -f '${remote_actor}/${f}'"; then
            rsync -av --progress "${NODE1_HOST}:${remote_actor}/${f}" "${actor_dst}/"
        fi
    done
}

verify_actor_dir() {
    local actor_dir="$1"
    local step_name="$2"

    [[ -d "$actor_dir" ]] || die "Missing actor dir: $actor_dir"

    local n missing
    n=$(count_model_shards "$actor_dir")
    if [[ "$n" != "$WORLD_SIZE" ]]; then
        missing=$(list_missing_ranks "$actor_dir")
        die "${step_name}: expected ${WORLD_SIZE} model shards, found ${n} in ${actor_dir}. Missing ranks: ${missing:-all}"
    fi

    [[ -f "$actor_dir/fsdp_config.json" ]] || die "${step_name}: missing fsdp_config.json"

    local ws
    ws=$(python3 -c "import json; print(json.load(open('${actor_dir}/fsdp_config.json'))['world_size'])" 2>/dev/null || echo "")
    if [[ -n "$ws" && "$ws" != "$WORLD_SIZE" ]]; then
        die "${step_name}: fsdp_config.json world_size=${ws}, expected ${WORLD_SIZE}"
    fi

    log "${step_name}: OK (${WORLD_SIZE} shards, fsdp_config.json present)"
}

gather_one_step() {
    local src0="$1"
    local step_name="$2"
    local actor_src="${src0}/${step_name}/actor"
    local actor_dst="${TARGET_ROOT}/${step_name}/actor"

    [[ -d "$actor_src" ]] || die "No actor checkpoint at ${actor_src}"

    mkdir -p "$actor_dst"

    # 1) Copy everything from local Ray dir (whichever ranks this node has)
    log "Sync ${step_name} from local -> ${TARGET_ROOT}"
    rsync -av --progress \
        "${src0}/${step_name}/" \
        "${TARGET_ROOT}/${step_name}/"

    # 2) Fill any missing ranks 0..15 from the other node (works whether local has 0-7 or 8-15)
    if [[ -n "$NODE1_HOST" ]]; then
        local remote_actor missing r
        remote_actor=$(remote_actor_dir "$step_name")
        log "Remote actor dir on ${NODE1_HOST}: ${remote_actor}"

        missing=$(list_missing_ranks "$actor_dst")
        if [[ -n "$missing" ]]; then
            log "Missing ranks on local, fetching from ${NODE1_HOST}: ${missing}"
            for r in $missing; do
                fetch_shard_from_remote "$remote_actor" "$actor_dst" "$r"
            done
        else
            log "All ${WORLD_SIZE} model shards already present after local sync"
        fi

        # fsdp_config.json / huggingface/ are written by rank 0 (usually on the other node)
        if [[ ! -f "$actor_dst/fsdp_config.json" ]]; then
            log "Fetching fsdp_config.json from ${NODE1_HOST}"
            rsync -av --progress "${NODE1_HOST}:${remote_actor}/fsdp_config.json" "${actor_dst}/"
        fi
        if [[ ! -d "$actor_dst/huggingface" ]] || [[ -z "$(ls -A "$actor_dst/huggingface" 2>/dev/null)" ]]; then
            log "Fetching huggingface/ from ${NODE1_HOST}"
            rsync -av --progress "${NODE1_HOST}:${remote_actor}/huggingface/" "${actor_dst}/huggingface/"
        fi
    else
        log "NODE1_HOST not set; assuming all shards already on this machine"
    fi

    verify_actor_dir "$actor_dst" "$step_name"
}

merge_one_step() {
    local step_name="$1"
    local actor_dir="${TARGET_ROOT}/${step_name}/actor"
    local hf_dir="${TARGET_ROOT}/${step_name}/${HF_SUFFIX}"

    verify_actor_dir "$actor_dir" "$step_name"

    mkdir -p "$hf_dir"
    log "Merging ${step_name} -> ${hf_dir}"
    cd "$VERL_ROOT"
    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$actor_dir" \
        --target_dir "$hf_dir"

    log "Done: ${hf_dir}"
}

# ── Main ──────────────────────────────────────────────────────────────────────
log "TARGET_ROOT=${TARGET_ROOT}"
log "VERL_ROOT=${VERL_ROOT}"
log "WORLD_SIZE=${WORLD_SIZE}"

SRC0=$(resolve_ray_src "local" "$RAY_SESSION_ROOT")
log "Local Ray checkpoint source: ${SRC0}"

mkdir -p "$TARGET_ROOT"

# Collect step list
steps=()
if [[ -n "$STEP" ]]; then
    steps+=("$STEP")
else
    for d in "$SRC0"/global_step_*; do
        [[ -d "$d" ]] || continue
        steps+=("$(basename "$d")")
    done
fi

if [[ ${#steps[@]} -eq 0 ]]; then
    die "No global_step_* found under ${SRC0}"
fi

log "Steps to process: ${steps[*]}"

# Phase 1: gather shards
for step_name in "${steps[@]}"; do
    gather_one_step "$SRC0" "$step_name"
done

if [[ "$GATHER_ONLY" == "1" ]]; then
    log "GATHER_ONLY=1, skipping model_merger"
    exit 0
fi

# Phase 2: merge to HF
for step_name in "${steps[@]}"; do
    merge_one_step "$step_name"
done

log "All done."
log "FSDP shards: ${TARGET_ROOT}/global_step_*/actor"
log "HF models:   ${TARGET_ROOT}/global_step_*/${HF_SUFFIX}"