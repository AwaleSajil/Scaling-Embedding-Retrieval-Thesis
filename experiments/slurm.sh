#!/bin/bash
#
#SBATCH --mail-user=sa0812@uah.edu
#SBATCH --job-name=e5_gamma_sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --time=14-0:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END,FAIL

export GPUS_PER_NODE=2
export OMP_NUM_THREADS=1
export HF_HUB_DISABLE_XET=1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

echo "===== SLURM ENVIRONMENT ====="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node List: $SLURM_NODELIST"
echo "CPUs:      $SLURM_CPUS_ON_NODE"
echo "MASTER_ADDR:PORT = $MASTER_ADDR:$MASTER_PORT"
echo "GPUS_PER_NODE = $GPUS_PER_NODE"
echo

echo "===== ACTIVATING CONDA ENV ====="
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate thesis
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
echo "===== STARTING TRAINING ====="

cd /rhome/sawale/thesis/experiments

echo "--- e5 gamma=0.05 (BAAI/bge-base-en-v1.5) ---"
srun --mem=0 torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
    --rdzv_backend=c10d \
    ../src/train.py --gradient_accumulation_steps 16 --expirement_number e5 --annealed_tanh_gamma 0.05
echo "--- e5 gamma=0.05 (bge) done ---"

MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
echo "--- e5 gamma=0.2 (BAAI/bge-base-en-v1.5) ---"
srun --mem=0 torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
    --rdzv_backend=c10d \
    ../src/train.py --gradient_accumulation_steps 16 --expirement_number e5 --annealed_tanh_gamma 0.2
echo "--- e5 gamma=0.2 (bge) done ---"

MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
echo "--- e5 gamma=0.05 (FacebookAI/roberta-base) ---"
srun --mem=0 torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
    --rdzv_backend=c10d \
    ../src/train.py --gradient_accumulation_steps 16 --expirement_number e5 --annealed_tanh_gamma 0.05 --model_name FacebookAI/roberta-base
echo "--- e5 gamma=0.05 (roberta) done ---"

MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
echo "--- e5 gamma=0.2 (FacebookAI/roberta-base) ---"
srun --mem=0 torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
    --rdzv_backend=c10d \
    ../src/train.py --gradient_accumulation_steps 16 --expirement_number e5 --annealed_tanh_gamma 0.2 --model_name FacebookAI/roberta-base
echo "--- e5 gamma=0.2 (roberta) done ---"

echo "===== TRAINING DONE ====="
