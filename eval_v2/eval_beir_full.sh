#!/bin/bash
#
#SBATCH --mail-user=sa0812@uah.edu
#SBATCH --job-name=beir_full_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --time=7-0:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END,FAIL

# ============================================================================
# Full BEIR evaluation for eval_v2 (all 12 subsets).
#
# Submit from the eval_v2/ directory (so slurm_logs/ resolves here):
#   cd /rhome/sawale/thesis/eval_v2 && mkdir -p slurm_logs && sbatch eval_beir_full.sh
#
# Resource notes:
#   * ~29 checkpoints to encode (training snapshots are excluded below).
#   * ~24.06M embeddings per checkpoint, 768-d float32 = ~74 GB cache each.
#   * Encode ~1.8 h/checkpoint on 3x A100; full run is multi-day -> --time 7d.
#   * 128 GB RAM is needed for the 5.4M-doc corpora (fever / climate-fever).
#   * Significance scales O(models^2); with ~224 variants it adds hours at the
#     end of the run. There is no skip flag yet -- ask if you want one added.
#
# All large artifacts go to the attached disk (set EXT_DISK below):
#   $EXT_DISK/hf_cache     BEIR raw data + base-model downloads (~50 GB, shared)
#   $EXT_DISK/eval_cache   float32 embedding cache (~74 GB per checkpoint)
#   $EXT_DISK/results      JSON + shards + HTML (~15-20 GB)
# ============================================================================

# ---- EDIT THIS: mount point of your attached / external disk -----------------
EXT_DISK=/rhome/sawale/extdisk        # <-- change to your actual mount
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
