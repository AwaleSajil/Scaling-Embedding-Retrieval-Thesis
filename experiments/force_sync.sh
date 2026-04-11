#!/bin/bash
# Force sync all local wandb runs to the new account.
# Usage: bash force_sync.sh <your_new_wandb_entity>
#
# Find your entity name by running: wandb whoami

ENTITY=${1?"Usage: bash force_sync.sh <your_wandb_entity>"}
PROJECT="scale_emb_retrieval"
WANDB_DIR="wandb"
MIN_SIZE_BYTES=1000  # skip stub runs (7-byte online-only runs)

echo "Syncing to entity='$ENTITY' project='$PROJECT'"
echo "---"

synced=0
skipped=0

for run_dir in "$WANDB_DIR"/run-*/; do
    wandb_file=$(find "$run_dir" -name "*.wandb" ! -name "*.synced" 2>/dev/null | head -1)

    if [ -z "$wandb_file" ]; then
        echo "SKIP (no .wandb file): $run_dir"
        ((skipped++))
        continue
    fi

    size=$(stat -c%s "$wandb_file" 2>/dev/null || echo 0)

    if [ "$size" -lt "$MIN_SIZE_BYTES" ]; then
        echo "SKIP (online-only stub, ${size}B): $run_dir"
        ((skipped++))
        continue
    fi

    echo "SYNCING (${size}B): $run_dir"
    wandb sync --no-mark-synced --include-online --include-synced --entity "$ENTITY" --project "$PROJECT" "$run_dir"
    ((synced++))
    echo "---"
done

echo ""
echo "Done. Attempted: $synced  |  Skipped (no local data): $skipped"
