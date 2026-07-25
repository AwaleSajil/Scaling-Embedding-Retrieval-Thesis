#!/bin/bash
#
#SBATCH --mail-user=sa0812@uah.edu
#SBATCH --job-name=beir_sub
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --output=slurm_logs/%A_%a_beir.out
#SBATCH --error=slurm_logs/%A_%a_beir.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-12%1
#SBATCH --mail-type=END,FAIL
#
# ============================================================================
# Full BEIR evaluation, one SLURM array task per subset.
#
#   cd /rhome/sawale/thesis/eval_v2 && mkdir -p slurm_logs
#   sbatch eval_beir_array.slurm
#   # when every task has finished:
#   sbatch eval_beir_finalize.slurm
#
# Why per subset instead of one job (eval_beir_full.sh):
#   * The whole run is ~6-7 days, against a 7-day wall clock. A single job that
#     is killed at the limit leaves no significance tests and no dashboard. Each
#     subset here is independently bounded -- msmarco, the worst, is ~2 days.
#   * Results are durable as each subset lands. store.already_evaluated() keys
#     on (model, subset), so a failed task can simply be resubmitted and will
#     skip what it already did.
#   * The small subsets finish first (ordering below), so there is something
#     worth looking at within hours rather than at the end of a week.
#
# %1 in the array spec runs ONE task at a time. matrix101 has 3 A100s and other
# users share it, so two concurrent 2-GPU tasks would not both schedule. Raise
# to %2 only if the node frees up.
#
# --skip_significance is what makes the split worthwhile: significance loads
# every model's per-query shards into memory and rewrites the whole
# significance.json (~3.8 GB at full scale). Doing that after each of 13 subsets
# would cost more than the evaluation. eval_beir_finalize.slurm does it once.
# ============================================================================

# Smallest corpus first, so quick results arrive early and any systematic
# problem (a subset that fails to load, an OOM) surfaces on a cheap task rather
# than 30 hours into msmarco.
SUBSETS=(
    nfcorpus            #     3,633 docs
    scifact             #     5,183
    arguana             #     8,674
    scidocs             #    25,657
    fiqa                #    57,638
    trec-covid          #   171,332
    webis-touche2020    #   382,545
    nq                  # 2,681,468
    dbpedia-entity      # 4,635,922
    hotpotqa            # 5,233,329
    fever               # 5,416,568
    climate-fever       # 5,416,593
    msmarco             # 8,841,823  (evaluated on the dev/validation qrels)
)
SUBSET="${SUBSETS[$SLURM_ARRAY_TASK_ID]}"

EXT_DISK=/nas/rgroup/dsig/llm-team/sajil-thesis

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_DISABLE_XET=1
export HF_HOME="$EXT_DISK/hf_cache"

echo "===== SLURM ENVIRONMENT ====="
echo "Array job:  $SLURM_ARRAY_JOB_ID  task $SLURM_ARRAY_TASK_ID"
echo "Subset:     $SUBSET"
echo "Node:       $SLURM_NODELIST"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate thesis_eval
echo "Python: $(which python)"
python -c "import torch, faiss, pyarrow; print('torch', torch.__version__, \
'| faiss', faiss.__version__, '| pyarrow', pyarrow.__version__, \
'| cuda', torch.cuda.is_available(), torch.cuda.device_count())"
echo

cd /rhome/sawale/thesis

# Every registered variant except training-snapshot checkpoints, which do not
# need full BEIR. 224 specs riding on 29 unique checkpoints, so the GPU cost is
# 29 encodes per subset, not 224.
MODELS=$(python - <<'PYCODE'
from eval_v2.config.models import MODELS
print(" ".join(k for k, s in MODELS.items() if "/checkpoints/checkpoint-" not in s.path))
PYCODE
)
echo "Evaluating $(echo $MODELS | wc -w) model variants on $SUBSET"
echo

mkdir -p "$EXT_DISK/eval_cache" "$EXT_DISK/results"

python -m eval_v2.run \
    --dataset beir \
    --subsets "$SUBSET" \
    --cache_dir  "$EXT_DISK/eval_cache/" \
    --output_dir "$EXT_DISK/results/" \
    --batch_size 128 \
    --skip_significance \
    --wandb --wandb_run_kind full \
    --wandb_name "beir-${SUBSET}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --models $MODELS

echo "===== SUBSET $SUBSET DONE ====="
