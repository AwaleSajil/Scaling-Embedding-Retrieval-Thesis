# Train
# torchrun --nproc_per_node=4 ../src/train.py  --expirement_number e5

# torchrun --nproc_per_node=4 ../src/train.py  --expirement_number e5 --model_name FacebookAI/roberta-base


# torchrun --nproc_per_node=4 ../src/train.py  --expirement_number e5 --model_name microsoft/mpnet-base


# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 ../src/train.py --expirement_number e5 --gradient_accumulation_steps 16 --model_name microsoft/mpnet-base



torchrun --nproc_per_node=4 ../src/train.py  --expirement_number e5 --model_name FacebookAI/roberta-base