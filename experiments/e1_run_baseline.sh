# Train
# torchrun --nproc_per_node=4 ../src/train.py --expirement_number e1

# torchrun --nproc_per_node=4 ../src/train.py --expirement_number e1 --model_name FacebookAI/roberta-base

# torchrun --nproc_per_node=4 ../src/train.py --expirement_number e1 --model_name FacebookAI/roberta-base --resume_checkpoint_path /rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260505_02-52-31/roberta-base/checkpoints/checkpoint-7000 --resume_run_id 1jahq5s6

# Evaluate
# python ../eval/eval.py --dataset_name nanobeir --json_output_path ../eval/results_json/ --output_dir_plots ../eval/results_plots/ --json_time_path ../eval/results_times/ --emb_info_path ../eval/results_emb_info/

# Store results