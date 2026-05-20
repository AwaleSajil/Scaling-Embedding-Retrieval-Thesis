# expirement detail

expirement_map = {
    "e1": "Baseline - Fine-tune on mixed dataset",
    "e2": "Add binarization layer (STE)",
    "e3": "MRL Training",
    "e4": "MRL Training + Binarization Layer (STE)",
    "e5": "Add annealed tanh binarization layer",
    <!-- "e6_1b": "Multi-bit annealed tanh quantization (1-bit)",
    "e6_2b": "Multi-bit annealed tanh quantization (2-bit)",
    "e6_3b": "Multi-bit annealed tanh quantization (3-bit)",
    "e6_4b": "Multi-bit annealed tanh quantization (4-bit)", -->
}

# for roberta base 

e1: https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/1jahq5s6

/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260505_02-52-31/roberta-base/final_model

e2: https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/koc92796

/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260505_22-10-20/roberta-base/final_model


e3: https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/50gzqgh0

puget 2
bash e4_run_bat_mrl.sh ; bash e5_run_atanh.sh ; bash e6_run_asigm.sh

e4:  https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/6voe57dg

/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260505_22-13-20/roberta-base/final_model

e5: https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/buwwyw9o

e6: https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/dkaxkg3p



# for microsoft/mpnet-base

e1: https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/e1pvxgyz
/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260509_14-40-29/mpnet-base/final_model


e2: https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/lop8zjjb
/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260509_21-03-33/mpnet-base/final_model


e3: https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/lwt3safi
/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260509_14-57-07/mpnet-base/final_model


e4: https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/iuzzso05
/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260509_15-00-14/mpnet-base/final_model


e5: puget 4 at tmux 0
https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/8r97x1ux
/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260509_15-06-08/mpnet-base/final_model (dont yet exist but will exist here once the training is complete)



# for experiment with annelaed tahnh fir different gamma value
## for bge base model
e5 - (y=0.05): https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/7wyomfu7

/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260510_11-25-29/bge-base-en-v1.5/final_model


e5 - (y=0.2): https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/owmi3etl

/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260510_17-18-22/bge-base-en-v1.5/final_model

## for roberta base model
e5 - (y=0.05): /rhome/sawale/thesis/experiments/slurm_logs/3867031_%x.err

https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/o6cs43yp

/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260510_23-13-40/roberta-base/final_model

e5 - (y=0.2): /rhome/sawale/thesis/experiments/slurm_logs/3867031_%x.err

https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/z8bsv5i7

/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260511_05-12-15/roberta-base/final_model

## for mpnet model
e5 - (y=0.05): /rhome/sawale/thesis/experiments/slurm_logs/3867032_%x.err
https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/t7u3a2j3
/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260510_11-25-29/mpnet-base/final_model

e5 - (y=0.2): /rhome/sawale/thesis/experiments/slurm_logs/3867032_%x.err
https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/dsfj07h4

/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260511_00-11-34/mpnet-base/final_model


# expirements with mrl + annealed tanh (after setting best gamma)
# no significant change, so selected gamma =0.1
## for bge base model
e5 - (y=0.1): 
https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/zfaigxhi
rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260512_00-41-40/bge-base-en-v1.5/final_model

## for roberta base model
e5 - (y=0.1): 
https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/ijit7wna
rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260512_12-38-06/roberta-base/final_model

## for mpnet model
e5 - (y=0.1): 
https://wandb.ai/sajil-awale-nasa-impact-uah/scale_emb_retrieval/runs/2hp3lhur
rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260512_19-41-28/mpnet-base/final_model