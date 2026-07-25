#!/bin/bash
#
#SBATCH --mail-user=sa0812@uah.edu
#SBATCH --job-name=beir_smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END,FAIL
#
# ============================================================================
# SLURM wrapper that runs eval_beir_smoke.sh on matrix -- the machine that will
# actually run eval_beir_full.sh.
#
#   cd /rhome/sawale/thesis/eval_v2 && mkdir -p slurm_logs && sbatch eval_beir_smoke.slurm
#
# Sized as a scaled-down rehearsal of the full run rather than a minimal test:
#   * 2 A100s, same as eval_beir_full.sh, so start_multi_process_pool actually
#     spawns 2 workers and the multi-GPU split is exercised. The 1-GPU version
#     never tested that path.
#   * 2 subsets (nfcorpus + scifact), so the cross-subset mean and the
#     per-subset HTML shard split are exercised -- both no-ops on one subset.
#   * 48 GB. Observed peak on the 1-GPU / 1-subset run was 4.6 GB; two GPU
#     workers plus the larger scifact corpus stay far under this. The full
#     script's 128 GB is sized for the 5.4M-doc fever / climate-fever PQ peak.
#   * 2 h instead of 7 d.
#
# SMOKE_DIR points at a MATRIX-SPECIFIC directory on purpose. The dsig1 run
# already populated $EXT_DISK/smoke, and that tree is on the same shared NAS --
# reusing it would turn every encode into a cache HIT and every model into
# "[Skip] already done", so the GPU path on matrix would never actually run.
# ============================================================================

export SMOKE_DIR=/nas/rgroup/dsig/llm-team/sajil-thesis/smoke_matrix

echo "===== SLURM ENVIRONMENT ====="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node List: $SLURM_NODELIST"
echo "CPUs:      $SLURM_CPUS_ON_NODE"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-<unset>}"   # set by --gres, honoured by the smoke script
echo

bash /rhome/sawale/thesis/eval_v2/eval_beir_smoke.sh
