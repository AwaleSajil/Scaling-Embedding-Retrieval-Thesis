#!/bin/bash
#
#SBATCH --mail-user=sa0812@uah.edu
#SBATCH --job-name=beir_full_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --time=7-0:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END,FAIL

# ============================================================================
# Full BEIR evaluation for eval_v2 (all 13 subsets).
#
# Submit from the eval_v2/ directory (so slurm_logs/ resolves here):
#   cd /rhome/sawale/thesis/eval_v2 && mkdir -p slurm_logs && sbatch eval_beir_full.sh
#
# Resource notes:
#   * 224 model variants ride on only 29 unique checkpoints. Every truncation,
#     binarization, INT8/INT4, PQ and TQ spec is a post-hoc transform of a cached
#     float32 embedding set, so the GPU cost is 29 encodes, not 224.
#   * ~32.91M embeddings per checkpoint, 768-d float32 = ~101 GB cache each,
#     2.9 TB in total. msmarco alone is 27 GB of that (8.84M docs) -- it was added
#     for parity with NanoMSMARCO and is evaluated on the BEIR-standard dev set
#     (6,980 queries) via BEIR_QRELS_SPLIT in config/datasets.py.
#   * fever and climate-fever share one corpus (verified identical modulo 25 docs)
#     but are still encoded twice, since cache.py keys on (model, dataset, subset).
#     Deduplicating them would cut ~450 GB.
#   * Only 2 of the node's 3 A100s are requested, so one stays free for other
#     users. Encode is ~2.5 h/checkpoint on 3x A100, so ~3.7 h on 2x; full run
#     is multi-day -> --time 7d.
#   * 128 GB RAM is needed for the 5.4M-doc corpora (fever / climate-fever).
#     The 90 PQ specs are the memory peak: sa_encode/sa_decode over a 5.4M x 768
#     corpus holds ~17 GB arrays (see evaluator._apply_pq).
#   * Per-query results are sharded one JSON per model under results/beir/
#     per_query/. A single per_query.json would hit ~12 GB (30,214 judged queries
#     x 224 models) and be rewritten after every (model, subset) pair.
#   * Significance scales O(models^2); with ~224 variants it adds hours at the
#     end of the run. There is no skip flag yet -- ask if you want one added.
#
# All large artifacts go to the attached disk (set EXT_DISK below):
#   $EXT_DISK/hf_cache     BEIR raw data + base-model downloads (~60 GB, shared)
#   $EXT_DISK/eval_cache   float32 embedding cache (~101 GB per checkpoint)
#   $EXT_DISK/results      JSON + shards + HTML (~20 GB)
# ============================================================================

# ---- Scratch space for the large artifacts -----------------------------------
# Group volume /nas/rgroup/dsig: 40 T total, ~20 T free. The 2.1 TB cache does
# not fit in $HOME (2 T quota, ~157 G free).
EXT_DISK=/nas/rgroup/dsig/llm-team/sajil-thesis
# -----------------------------------------------------------------------------

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # speeds up the CPU-side scoring
export HF_HUB_DISABLE_XET=1
export HF_HOME="$EXT_DISK/hf_cache"           # BEIR data + base model downloads

echo "===== SLURM ENVIRONMENT ====="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node List: $SLURM_NODELIST"
echo "CPUs:      $SLURM_CPUS_ON_NODE"
echo "EXT_DISK:  $EXT_DISK"
echo "HF_HOME:   $HF_HOME"
echo

echo "===== ACTIVATING CONDA ENV ====="
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate thesis_eval
# thesis_eval = clone of the training env + eval-only deps (faiss-cpu, scipy,
# turboquant). See eval_v2/SETUP.md to create it. To run against the plain
# training env instead, change this to "conda activate thesis".
echo "Python: $(which python)"
python --version
echo

echo "===== TORCH GPU TEST ====="
python - << 'PYCODE'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"Number of GPUs: {n}")
    for i in range(n):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
PYCODE
echo

cd /rhome/sawale/thesis

# Build the model list: every registered variant EXCEPT the training-snapshot
# checkpoints (paths containing /checkpoints/checkpoint-), which do not need
# full BEIR. Variants that share a checkpoint reuse a single embedding cache.
echo "===== SELECTING MODELS ====="
MODELS=$(python - << 'PYCODE'
from eval_v2.config.models import MODELS
keys = [k for k, s in MODELS.items() if "/checkpoints/checkpoint-" not in s.path]
print(" ".join(keys))
PYCODE
)
echo "Evaluating $(echo $MODELS | wc -w) model variants"
echo

echo "===== STARTING FULL BEIR EVAL ====="
mkdir -p "$EXT_DISK/eval_cache" "$EXT_DISK/results"

python -m eval_v2.run \
    --dataset beir \
    --cache_dir  "$EXT_DISK/eval_cache/" \
    --output_dir "$EXT_DISK/results/" \
    --batch_size 128 \
    --models $MODELS

echo "===== BEIR EVAL DONE ====="
