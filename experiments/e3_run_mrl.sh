# Train
torchrun --nproc_per_node=4 ../src/train.py  --expirement_number e3

# Evaluate
python ../eval/eval.py --dataset_name nanobeir
# Store results