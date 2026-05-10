# Train
# torchrun --nproc_per_node=4 ../src/train.py  --expirement_number e2

torchrun --nproc_per_node=4 ../src/train.py --expirement_number e2 --model_name FacebookAI/roberta-base

# Evaluate
# python ../eval/eval.py --dataset_name nanobeir --json_output_path ../eval/results_json/ --output_dir_plots ../eval/results_plots/ --json_time_path ../eval/results_times/ --emb_info_path ../eval/results_emb_info/

# Store results