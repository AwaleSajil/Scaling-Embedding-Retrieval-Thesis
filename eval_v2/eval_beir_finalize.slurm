#!/bin/bash
#
#SBATCH --mail-user=sa0812@uah.edu
#SBATCH --job-name=beir_final
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --output=slurm_logs/%j_beir_finalize.out
#SBATCH --error=slurm_logs/%j_beir_finalize.err
#SBATCH --time=7-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END,FAIL
#
# ============================================================================
# Closes out a split BEIR run: significance tests, dashboard, HF deploy.
# Run once every task of eval_beir_array.slurm has finished.
#
#   cd /rhome/sawale/thesis/eval_v2 && sbatch eval_beir_finalize.slurm
#
# Or chain it so it fires automatically when the array completes:
#   AID=$(sbatch --parsable eval_beir_array.slurm)
#   sbatch --dependency=afterok:$AID eval_beir_finalize.slurm
#
# No GPU: significance and HTML building are pure CPU. Asking for one would
# queue behind GPU work for no reason.
#
# --finalize skips the evaluation loop entirely. A plain re-run would reload
# every dataset and call get_or_compute for all 29 checkpoints -- ~2.9 TB of
# NAS reads -- before skipping the already-evaluated pairs.
#
# 200 GB of RAM because compute_significance() calls load_per_query(), which
# merges EVERY model's per-query shard into one dict. At 224 models x 30,214
# queries that is ~7 GB on disk and several times that as live Python objects.
# This is the memory high-water mark of the whole pipeline, and it happens last,
# after days of successful work -- so it is over-provisioned deliberately.
#
# --time is 7 days against an estimate of 4-12 h, for the same reason: this job
# runs after every subset has completed, so being killed here would waste the
# entire run's tail. Significance is the least-measured stage (~5.6M paired
# tests, plus Python win/tie/loss loops over every query), and it is also the
# one most likely to be slower than projected.
# ============================================================================

EXT_DISK=/nas/rgroup/dsig/llm-team/sajil-thesis

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_DISABLE_XET=1
export HF_HOME="$EXT_DISK/hf_cache"

# Set to deploy the dashboard when finalizing; leave unset to skip.
HF_REPO="${HF_REPO:-}"

echo "===== SLURM ENVIRONMENT ====="
echo "Job ID: $SLURM_JOB_ID   Node: $SLURM_NODELIST   CPUs: $SLURM_CPUS_ON_NODE"
echo

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate thesis_eval
cd /rhome/sawale/thesis

# Report what actually completed before spending hours on significance over it.
echo "===== COVERAGE ====="
python -m eval_v2.watch_progress \
    --results_dir "$EXT_DISK/results/beir" \
    --cache_dir   "$EXT_DISK/eval_cache"
echo

echo "===== SIGNIFICANCE + DASHBOARD ====="
python -m eval_v2.run \
    --dataset beir \
    --output_dir "$EXT_DISK/results/" \
    --finalize \
    --wandb --wandb_run_kind full \
    --wandb_name "beir-finalize-${SLURM_JOB_ID}"

if [[ -n "$HF_REPO" ]]; then
    echo
    echo "===== HUGGINGFACE DEPLOY ====="
    python -m eval_v2.hf_deploy_dashboard \
        --results_dir "$EXT_DISK/results/beir" \
        --repo_id "$HF_REPO" \
        --commit_message "Full BEIR evaluation"
fi

echo "===== FINALIZE DONE ====="
