#!/bin/bash
set -euo pipefail
#
# ============================================================================
# Smoke test for the full-BEIR eval path -- proves the code works before
# burning multi-day A100 time on eval_beir_full.sh.
#
# Runs on either machine:
#   dsig1 (no SLURM):  cd /rhome/sawale/thesis && bash eval_v2/eval_beir_smoke.sh
#   matrix (SLURM):    sbatch eval_v2/eval_beir_smoke.slurm
#
# Env vars honoured:
#   SMOKE_DIR             output root (default $EXT_DISK/smoke). Point this at a
#                         fresh dir to force re-encoding instead of cache hits.
#   SMOKE_PY              explicit python to use; otherwise auto-detected below.
#   CUDA_VISIBLE_DEVICES  respected if already set (SLURM sets it).
#
# What makes it small:
#   * The two smallest BEIR subsets: nfcorpus (3,633 docs / 323 judged test
#     queries) and scifact (5,183 / 300). Next up is arguana at 8,674 and then
#     scidocs at 25,657, so these two are the cheapest pair that still exercises
#     the real BEIR loader -- and two subsets make the cross-subset mean and the
#     per-subset shard split meaningful, which a single subset cannot.
#   * 9 model specs riding on 5 checkpoints, chosen so every branch of the
#     transform pipeline runs at least once: float cosine, hamming/binary,
#     INT8, PQ, TurboQuant, MRL truncation, annealed-tanh wrapper, and the
#     multi-bit annealed-sigmoid wrapper.
#   * Significance is 9^2 pairs instead of 224^2, so it finishes in seconds.
#
# What it does NOT cover (too big to smoke-test; verified separately):
#   * msmarco's BEIR_QRELS_SPLIT="validation" override -- see the qrels-only
#     check at the bottom of this script.
#   * The 128 GB memory peak on fever / climate-fever.
#
# Output goes to a throwaway smoke/ tree so the real run's cache and results
# are never touched. HF_HOME is shared with the real run on purpose, so the
# downloaded nfcorpus data is reused later instead of fetched twice.
# ============================================================================

EXT_DISK=/nas/rgroup/dsig/llm-team/sajil-thesis
SMOKE_DIR="${SMOKE_DIR:-$EXT_DISK/smoke}"

export HF_HUB_DISABLE_XET=1
export HF_HOME="$EXT_DISK/hf_cache"           # shared with the real run
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

cd /rhome/sawale/thesis

echo "===== ENVIRONMENT ====="
echo "Host:        $(hostname)"
echo "SMOKE_DIR:   $SMOKE_DIR"
echo "HF_HOME:     $HF_HOME"

# ---- Interpreter selection ---------------------------------------------------
# The two machines need DIFFERENT environments and neither works on the other:
#   dsig1  = RHEL 9 / glibc 2.34 -> the uv .venv (torch 2.8.0+cu128) runs fine.
#   matrix = CentOS 7.9 / glibc 2.17, libstdc++ tops out at CXXABI_1.3.7. The
#            .venv dies with "CXXABI_1.3.9 not found" importing numpy, and torch
#            2.8 needs glibc >= 2.28 regardless. Must use the thesis_eval conda
#            env, whose conda-forge builds bundle their own libstdc++ (this is
#            exactly the trap SETUP.md's "Old server notes" warns about).
# Probe rather than branch on hostname, so the fallback is self-correcting.
CONDA_EVAL_PY="$HOME/miniconda3/envs/thesis_eval/bin/python"
VENV_PY=/rhome/sawale/thesis/.venv/bin/python
DEPS='import numpy, faiss, scipy, turboquant, torch'

# faiss in thesis_eval is a pip wheel (faiss-cpu 1.9.0.post1), not conda-forge:
# conda-forge's faiss is unsatisfiable against this defaults-channel env
# (_libgcc_mutex main vs conda_forge). 1.9.0.post1 is the LAST release with a
# cp312 manylinux2014 wheel -- anything newer is manylinux_2_28 and pip falls
# back to an sdist that tries to build numpy from source and fails.
# It imports against CentOS 7's system libstdc++, but fall back to the env's own
# lib dir (libstdcxx-ng 11.2 = GLIBCXX_3.4.29) if a future build needs it.
CONDA_LIB="$HOME/miniconda3/envs/thesis_eval/lib"

if [[ -n "${SMOKE_PY:-}" ]]; then
    PY="$SMOKE_PY"
elif "$VENV_PY" -c "$DEPS" >/dev/null 2>&1; then
    PY="$VENV_PY"
elif [[ -x "$CONDA_EVAL_PY" ]] && "$CONDA_EVAL_PY" -c "$DEPS" >/dev/null 2>&1; then
    PY="$CONDA_EVAL_PY"
elif [[ -x "$CONDA_EVAL_PY" ]] && \
     LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}" "$CONDA_EVAL_PY" -c "$DEPS" >/dev/null 2>&1; then
    PY="$CONDA_EVAL_PY"
    export LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}"
    echo "Note:        using env libstdc++ via LD_LIBRARY_PATH"
else
    echo "ERROR: no usable interpreter." >&2
    echo "  tried $VENV_PY and $CONDA_EVAL_PY" >&2
    echo "  On matrix, create thesis_eval per eval_v2/SETUP.md." >&2
    exit 1
fi
export PATH="$(dirname "$PY"):$PATH"
python() { "$PY" "$@"; }          # every `python` below goes through $PY

echo "Python:      $PY"
python --version
python -c "import faiss, scipy, turboquant, torch; \
print('faiss', faiss.__version__, '| torch', torch.__version__, \
'| cuda', torch.cuda.is_available(), torch.cuda.device_count())"

# ---- GPU selection ----------------------------------------------------------
# dsig1 is heterogeneous: 1x V100 (sm_70) + 2x P100 (sm_60). torch 2.8.0+cu128
# ships kernels for sm_70..sm_120 only, so the moment start_multi_process_pool
# puts a worker on a P100 the encode dies with
#   torch.AcceleratorError: CUDA error: no kernel image is available ...
# Select only devices whose compute capability is in torch's arch list. matrix's
# A100s are sm_80, so eval_beir_full.sh does not need this.
# Override manually with e.g. CUDA_VISIBLE_DEVICES=0 bash eval_v2/eval_beir_smoke.sh
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES=$(python - 2>/dev/null <<'PYCODE'
import torch
archs = {a[3:] for a in torch.cuda.get_arch_list() if a.startswith("sm_")}
usable = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    if f"{p.major}{p.minor}" in archs:
        usable.append(str(i))
print(",".join(usable))
PYCODE
)
fi
export CUDA_VISIBLE_DEVICES
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    echo "ERROR: no GPU on this host matches torch's supported archs." >&2
    exit 1
fi
echo "GPUs:        $CUDA_VISIBLE_DEVICES (supported archs only)"
echo

# One spec per transform branch. Keep this list in sync with the branches in
# eval_v2/wrappers/transforms.py and evaluator._apply_pq / _apply_tq.
MODELS=(
    bge-base-en-v1.5                          # float cosine, HF-hub checkpoint
    finetuned-bge-base-en-v1.5                # float cosine, local checkpoint
    finetuned-bge-base-en-v1.5-post-binary    # binarize -> uint8 packed, hamming
    finetuned-bge-base-en-v1.5-post-int8      # INT8 quantize
    finetuned-bge-base-en-v1.5-pq-m16-n8      # PQ reconstruct (the del/no-add change)
    finetuned-bge-base-en-v1.5-tq-8bit        # TurboQuant dequantize
    finetuned-bge-base-en-v1.5-mrl-32         # truncate_dim
    finetuned-bge-base-en-v1.5-atanh-gamma0.1 # AnnealedTanh layer at load time
    finetuned-bge-base-en-v1.5-asigm-2b       # multi-bit annealed sigmoid
)

# The two smallest BEIR subsets. Override with e.g. SMOKE_SUBSETS="nfcorpus" to
# go faster, or add scidocs/arguana to go bigger.
read -r -a SUBSETS <<< "${SMOKE_SUBSETS:-nfcorpus scifact}"

echo "===== RUNNING SMOKE EVAL (beir / ${SUBSETS[*]}, ${#MODELS[@]} specs) ====="
mkdir -p "$SMOKE_DIR/eval_cache" "$SMOKE_DIR/results"

# Set SMOKE_WANDB=1 to report progress to W&B. The run is tagged "smoke" so it
# filters out of the real runs; see eval_v2/monitor.py.
RUN_FLAGS=()
[[ -n "${SMOKE_WANDB:-}" ]] && RUN_FLAGS+=(--wandb --wandb_run_kind smoke)

python -m eval_v2.run \
    --dataset beir \
    --subsets "${SUBSETS[@]}" \
    --cache_dir  "$SMOKE_DIR/eval_cache/" \
    --output_dir "$SMOKE_DIR/results/" \
    --batch_size 128 \
    --models "${MODELS[@]}" \
    "${RUN_FLAGS[@]}"

echo
echo "===== CHECKING OUTPUTS ====="
# run.py nests everything one level deeper, under <output_dir>/<dataset>/
export SMOKE_OUT="$SMOKE_DIR/results/beir"
export SMOKE_SUBSETS_CHECK="${SUBSETS[*]}"
python - <<'PYCODE'
import json, os, sys
from pathlib import Path

out = Path(os.environ["SMOKE_OUT"])
wanted = os.environ["SMOKE_SUBSETS_CHECK"].split()
problems = []

agg = json.loads((out / "aggregate.json").read_text())
print(f"aggregate.json: {len(agg)} models x {len(wanted)} subsets")
hdr = "".join(f"{s:>14s}" for s in wanted)
print(f"  {'model':45s}{hdr}      mean")
for mkey, subsets in sorted(agg.items()):
    vals = [subsets.get(s, {}).get("ndcg@10") for s in wanted]
    cells = "".join(f"{v:14.4f}" if v else f"{'MISSING':>14s}" for v in vals)
    # run.py writes a "mean" pseudo-subset after each run; report it if present
    mean = subsets.get("mean", {}).get("ndcg@10")
    print(f"  {mkey:45s}{cells}  {mean:8.4f}" if mean else f"  {mkey:45s}{cells}       n/a")
    for s, v in zip(wanted, vals):
        if not v:
            problems.append(f"{mkey}/{s}: ndcg@10 is {v!r} "
                            f"(all-zero metrics usually means qrels IDs did not match)")

# per_query must be sharded one file per model, not one monolithic JSON.
# One file per MODEL regardless of subset count -- subsets are keys inside it.
shards = sorted((out / "per_query").glob("*.json")) if (out / "per_query").is_dir() else []
print(f"\nper_query/ shards: {len(shards)}")
if len(shards) != len(agg):
    problems.append(f"expected {len(agg)} per_query shards, found {len(shards)}")
for sh in shards:
    have = set(json.loads(sh.read_text()))
    missing = set(wanted) - have
    if missing:
        problems.append(f"per_query/{sh.name} missing subsets: {sorted(missing)}")

# the HTML builder shards per SUBSET, so this count should track the subset list
html_shards = sorted((out / "per_query_shards").glob("*.json")) \
    if (out / "per_query_shards").is_dir() else []
print(f"per_query_shards/ (by subset): {len(html_shards)}")
if len(html_shards) != len(wanted):
    problems.append(f"expected {len(wanted)} per-subset HTML shards, found {len(html_shards)}")
if (out / "per_query.json").exists():
    problems.append("a monolithic per_query.json was written -- sharding did not take effect")

sig = json.loads((out / "significance.json").read_text()) if (out / "significance.json").exists() else {}
print(f"significance.json: {len(sig)} pairs")
if not sig:
    problems.append("significance.json is empty")

html = out / "index.html"
print(f"index.html: {'%.1f MB' % (html.stat().st_size / 1e6) if html.exists() else 'MISSING'}")
if not html.exists():
    problems.append("index.html was not built")

if problems:
    print("\nFAILED:")
    for p in problems:
        print("  *", p)
    sys.exit(1)
print("\nAll checks passed.")
PYCODE

echo
echo "===== msmarco QRELS SPLIT CHECK (qrels only, no encoding) ====="
# The one piece the smoke run cannot cover: BEIR_QRELS_SPLIT sends msmarco to
# the dev/validation split. Loading just the qrels repo is a few MB.
python - <<'PYCODE'
from datasets import load_dataset
from eval_v2.config.datasets import BEIR_QRELS_SPLIT

split = BEIR_QRELS_SPLIT.get("msmarco", "test")
ds = load_dataset("BeIR/msmarco-qrels", split=split)
qids = {r["query-id"] for r in ds}
print(f"split={split}  judgments={len(ds)}  unique queries={len(qids)}")
assert split == "validation", f"expected validation, got {split}"
assert len(qids) > 6000, f"only {len(qids)} queries -- wrong split (TREC-DL test has 43)"
print("msmarco split override OK")
PYCODE

echo
echo "===== HUGGINGFACE DEPLOY ====="
# Only runs when SMOKE_HF_REPO is set, so a plain smoke run never publishes
# anything. Set SMOKE_HF_PRIVATE=1 to create the Space private, and
# SMOKE_HF_DRYRUN=1 to list what would upload without creating the repo.
if [[ -n "${SMOKE_HF_REPO:-}" ]]; then
    DEPLOY_FLAGS=()
    [[ -n "${SMOKE_HF_PRIVATE:-}" ]] && DEPLOY_FLAGS+=(--private)
    [[ -n "${SMOKE_HF_DRYRUN:-}" ]] && DEPLOY_FLAGS+=(--dry_run)
    python -m eval_v2.hf_deploy_dashboard \
        --results_dir "$SMOKE_OUT" \
        --repo_id "$SMOKE_HF_REPO" \
        --commit_message "BEIR smoke test (nfcorpus, ${#MODELS[@]} specs)" \
        "${DEPLOY_FLAGS[@]}"
else
    echo "skipped (set SMOKE_HF_REPO=<user>/<space> to deploy)"
fi

echo
echo "===== SMOKE TEST DONE ====="
echo "Dashboard: $SMOKE_DIR/results/beir/index.html"
echo "Delete when finished:  rm -rf $SMOKE_DIR"
