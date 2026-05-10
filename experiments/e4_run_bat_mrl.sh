# Train
# torchrun --nproc_per_node=4 ../src/train.py  --expirement_number e4

# torchrun --nproc_per_node=4 ../src/train.py --expirement_number e4 --model_name FacebookAI/roberta-base

torchrun --nproc_per_node=4 ../src/train.py --expirement_number e4 --model_name microsoft/mpnet-base

# Evaluate
# python ../eval/eval.py --dataset_name nanobeir --json_output_path ../eval/results_json/ --output_dir_plots ../eval/results_plots/ --json_time_path ../eval/results_times/ --emb_info_path ../eval/results_emb_info/
