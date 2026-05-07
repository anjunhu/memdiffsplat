#!/usr/bin/env bash
# Launch diffsplat-memeval unlearning runners in tmux sessions on g6e.
# - Creates named tmux sessions if they don't exist (skips if already running)
# - Only launches on GPUs with >= 40000 MiB free VRAM
# - Activates conda env 'diffsplat'
# Usage: bash tmux_launch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# ── GPU availability check ───────────────────────────────────────────────────
MIN_FREE_MIB=40000

CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free \
    --format=csv,noheader,nounits 2>/dev/null \
  | awk -F',' -v threshold="$MIN_FREE_MIB" \
      '$2 >= threshold {printf "%s%s", (NR==1?"":sep), $1; sep=","}')

if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
  echo "No GPU with >= ${MIN_FREE_MIB} MiB free found. Aborting."
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader 2>/dev/null || true
  exit 1
fi

export CUDA_VISIBLE_DEVICES
echo "GPU check passed — using devices: $CUDA_VISIBLE_DEVICES"

IFS=',' read -ra GPU_POOL <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT=${#GPU_POOL[@]}
GPU_INDEX=0

# ── Helper ───────────────────────────────────────────────────────────────────
launch() {
  local session="$1"
  local cmd="$2"

  if [[ $GPU_INDEX -ge $GPU_COUNT ]]; then
    echo "  $session: no GPU available — retry after current runs finish"
    return 0
  fi
  local assigned_gpu="${GPU_POOL[$GPU_INDEX]}"
  GPU_INDEX=$(( GPU_INDEX + 1 ))

  if tmux has-session -t "$session" 2>/dev/null; then
    local pane_pid py_pid
    pane_pid=$(tmux list-panes -t "$session" -F '#{pane_pid}' 2>/dev/null | head -1)
    py_pid=$(pstree -p "$pane_pid" 2>/dev/null | grep -oP 'python3?\(\K[0-9]+' | head -1 || true)
    if [[ -n "$py_pid" ]]; then
      echo "  $session: already running (pid=$py_pid) — skipped"
      GPU_INDEX=$(( GPU_INDEX - 1 ))
      return 0
    fi
    tmux kill-session -t "$session" 2>/dev/null || true
  fi

  tmux new-session -d -s "$session"
  sleep 0.5
  tmux send-keys -t "$session" \
    "export CUDA_VISIBLE_DEVICES=$assigned_gpu && conda activate diffsplat && cd $PROJECT_ROOT && $cmd 2>&1 | tee /tmp/${session}.log" \
    Enter
  echo "  $session: launched — GPU $assigned_gpu"
}

# ── Unlearning runners ────────────────────────────────────────────────────────
echo "=== Launching diffsplat-memeval unlearning runners ==="

# launch "ds-nemo"            "python runners/run_nemo.py"
launch "ds-sail"            "python runners/run_sail.py"
launch "ds-amg"             "python runners/run_amg.py"
launch "ds-subspace-prune"  "python runners/run_subspace_prune.py"
launch "ds-uce-fixed"       "python runners/run_uce_fixed_concept.py"
launch "ds-uce-multi"       "python runners/run_uce_multi_concept.py"
launch "ds-xattn-be"        "python runners/run_xattn.py --method be"
launch "ds-xattn-ca-ent"    "python runners/run_xattn.py --method ca_entropy"
launch "ds-ip-rt"           "python runners/run_input_perturb.py --method ip-rt"
launch "ds-ip-rna"          "python runners/run_input_perturb.py --method ip-rna"
launch "ds-ip-cwr"          "python runners/run_input_perturb.py --method ip-cwr"
launch "ds-ip-gni"          "python runners/run_input_perturb.py --method ip-gni"
launch "ds-ip-wen"          "python runners/run_input_perturb.py --method ip-wen"

echo "=== Done ==="
