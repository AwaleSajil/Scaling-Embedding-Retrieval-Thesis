import argparse
import datetime
import inspect
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Union
import sys
import torch
import wandb
import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets
from datasets import config as dataset_config
from datasets import get_dataset_config_names, load_dataset, load_from_disk
from utils.distributed import init_ddp, print0
from utils import distributed
from dotenv import load_dotenv
from utils.multidataset_sampler import WeightedBatchSampler
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    models,
    util,
)
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
    TripletEvaluator,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.sampler import (
    MultiDatasetDefaultBatchSampler,
    ProportionalBatchSampler,
    RoundRobinBatchSampler,
)
from sentence_transformers.training_args import (
    BatchSamplers,
    MultiDatasetBatchSamplers,
    SentenceTransformerTrainingArguments,
)
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import BatchSampler, ConcatDataset
from transformers import TrainerCallback
from transformers.optimization import get_scheduler  # For fallback in custom trainer
from utils.utils import (
    AnnealedTanhBinarizationLayer,
    BinarizationLayer,
    MultiBitAnnealedSigmoidLayer,
    PreTokenizedCollator,
    build_dataset_configs,
    get_gpu_info,
    hamming_sim,
    load_and_cache_datasets,
    prepare_evaluators,
)

_THESIS_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_THESIS_ROOT / ".env", override=True)

# Set HF_TOKEN from HUGGINGFACE_TOKEN so all HF libraries pick it up automatically.
# We do NOT call login() here — it writes to a shared file and races under DDP.
_hf_token = os.getenv("HUGGINGFACE_TOKEN")
if _hf_token:
    os.environ["HF_TOKEN"] = _hf_token
# ──────────────── Constants ────────────────

parser = argparse.ArgumentParser(description="Sentence Transformer Training Config")

parser.add_argument(
    "--config",
    type=str,
    default="../src/config.yaml",
    help="Path to the config file",
)
parser.add_argument(
    "--nrows", 
    type=int, 
    help="Number of datapoints to use per source for training")
parser.add_argument(
    "--n_data_src",
    type=int,
    help="Number of data sources to use for training",
)
parser.add_argument(
    "--model_max_len", 
    type=int, 
    help="Maximum sequence length for the model"
    )
parser.add_argument(
    "--model_name", 
    type=str, 
    help="Name of the model to use for training"
    )
parser.add_argument(
    "--output_base", 
    type=str, 
    help="Base directory for output models"
    )
parser.add_argument(
    "--wb_mode",
    type=str,
    choices=["online", "offline", "disabled"],
)
parser.add_argument(
    "--wb_project",
    type=str,
    choices=["scale_emb_retrieval"],
)
parser.add_argument(
    "--resume_checkpoint_path",
    type=str,
    help="Path to the checkpoint to resume training from"
    )
parser.add_argument(
    "--resume_run_id", 
    type=str, 
    help="WB run ID to resume training from"
    )
parser.add_argument(
    "--num_train_epochs", 
    type=int, 
    help="Number of training epochs"
    )
parser.add_argument(
    "--batch_size", 
    type=int, 
    help="Batch size for training"
    )
parser.add_argument(
    "--eval_and_save_steps", 
    type=int, 
    default=1000
    )
parser.add_argument(
    "--gradient_accumulation_steps", 
    type=int, 
    default=8
    )
parser.add_argument(
    "--max_datapoints_per_src_for_eval", 
    type=int, 
    default=20
    )
parser.add_argument(
    "--lr", 
    type=float, 
    default=2e-5
    )
parser.add_argument(
    "--expirement_number",
    type=str,
    default="e1",
    choices=["e1", "e2", "e3", "e4", "e5", "e6_1b", "e6_2b", "e6_3b", "e6_4b", "e7"]
    )
parser.add_argument(
    "--annealed_tanh_gamma",
    type=float,
    default=None,
    help="Override annealed_tanh_config.gamma from config.yaml (e5/e6 only)",
    )

expirement_map = {
    "e1": "Baseline - Fine-tune on mixed dataset",
    "e2": "Add binarization layer (STE)",
    "e3": "MRL Training",
    "e4": "MRL Training + Binarization Layer (STE)",
    "e5": "Add annealed tanh binarization layer",
    "e7": "MRL Training + Annealed Tanh Binarization",
    "e6_1b": "Multi-bit annealed tanh quantization (1-bit)",
    "e6_2b": "Multi-bit annealed tanh quantization (2-bit)",
    "e6_3b": "Multi-bit annealed tanh quantization (3-bit)",
    "e6_4b": "Multi-bit annealed tanh quantization (4-bit)",
}

args = parser.parse_args()


# load the config.yaml file
with open(args.config, "r") as f:
    config_default = yaml.safe_load(f)

# Check if bf16 and fp16 are supported
bf16_supported = torch.cuda.is_bf16_supported()
fp16_supported = torch.cuda.is_available()

# defining the main config dict
# if the config is set in terminal args, use it, otherwise use the default from config.yaml

# defining the main config dict
# if the config is set in terminal args, use it, otherwise use the default from config.yaml
config = {
    "input_data": {
        "nrows": args.nrows or config_default.get("input_data").get("nrows"),
        "n_data_src": args.n_data_src or config_default.get("input_data").get("n_data_src"),
        "max_datapoints_per_src_for_eval": args.max_datapoints_per_src_for_eval or config_default.get("input_data").get("max_datapoints_per_src_for_eval"),
    },
    "input_model": {
        "name": args.model_name or config_default.get("input_model").get("name"),
        "max_len": args.model_max_len or config_default.get("input_model").get("max_len"),
    },
    "outputs": {
        "output_base": args.output_base or config_default.get("outputs").get("output_base"),
        "cache_dir": f"{config_default.get('outputs').get('cache_dir')}/NROWS_{args.nrows or config_default.get('input_data').get('nrows')}",
    },
    "resume_config": {
        "resume_checkpoint_path": args.resume_checkpoint_path or config_default.get("resume_config").get("resume_checkpoint_path"),
        "resume_run_id": args.resume_run_id or config_default.get("resume_config").get("resume_run_id"),
    },
    "trainer_config": {
        "num_train_epochs": args.num_train_epochs or config_default.get("trainer_config").get("num_train_epochs"),
        "per_device_train_batch_size": args.batch_size or config_default.get("trainer_config").get("per_device_train_batch_size"),
        "per_device_eval_batch_size": args.batch_size or config_default.get("trainer_config").get("per_device_eval_batch_size"),
        "warmup_ratio": config_default.get("trainer_config").get("warmup_ratio"),
        "fp16": not bf16_supported and fp16_supported,
        "bf16": bf16_supported,
        "batch_sampler": BatchSamplers.NO_DUPLICATES,
        "multi_dataset_batch_sampler": WeightedBatchSampler,
        "eval_strategy": None, # initilize based on gpu rank
        "eval_steps": args.eval_and_save_steps or config_default.get("trainer_config").get("eval_and_save_steps"),
        "save_strategy": config_default.get("trainer_config").get("save_strategy"),
        "save_steps": args.eval_and_save_steps or config_default.get("trainer_config").get("eval_and_save_steps"),
        "save_total_limit": config_default.get("trainer_config").get("save_total_limit"),
        "logging_steps": args.eval_and_save_steps or config_default.get("trainer_config").get("eval_and_save_steps"),
        "learning_rate": args.lr or config_default.get("trainer_config").get("learning_rate"),
        "lr_scheduler_type": config_default.get("trainer_config").get("lr_scheduler_type"),
        "gradient_accumulation_steps": args.gradient_accumulation_steps or config_default.get("trainer_config").get("gradient_accumulation_steps"),
        "weight_decay": config_default.get("trainer_config").get("weight_decay"),
        "report_to": config_default.get("trainer_config").get("report_to"),
        "local_rank": None, # initillize later
        "ignore_data_skip": config_default.get("trainer_config").get("ignore_data_skip")

    },
    "logging": {
        "wb_mode": args.wb_mode or config_default.get("logging").get("wb_mode"),
        "wb_project": args.wb_project or config_default.get("logging").get("wb_project"),
    },
    "mrl_config": {
        "matryoshka_dims": config_default.get("mrl_config").get("matryoshka_dims"),
    },
    "annealed_tanh_config": {
        "gamma": args.annealed_tanh_gamma if args.annealed_tanh_gamma is not None else config_default.get("annealed_tanh_config").get("gamma"),
    },
    "multibit_asigmoid_config": {
        "gamma":        config_default.get("multibit_asigmoid_config", {}).get("gamma", 0.1),
        "init_scale":   config_default.get("multibit_asigmoid_config", {}).get("init_scale", 0.036),
        "warmup_steps": config_default.get("multibit_asigmoid_config", {}).get("warmup_steps", 500),
    },
    "experiment": {
        "number": args.expirement_number,
        "desc": expirement_map.get(args.expirement_number),
    },
}



current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y%m%d_%H-%M-%S")
os.makedirs(config["outputs"]["cache_dir"], exist_ok=True)
# assert os.getenv("WANDB_LOG_MODEL") == "end"

# HF auth is handled via HF_TOKEN env var set above — no login() call needed.


# Set dataset config load up to 800gb into the memory for faster training speed
dataset_config.IN_MEMORY_MAX_SIZE = 800 * (1024**3)



# ──────────────── Custom Classes ────────────────
class CustomSentenceTransformerTrainer(SentenceTransformerTrainer):
    def __init__(
        self,
        *args_trainer,
        custom_lr_params: Dict = None,
        dataset_configs=None,
        **kwargs_trainer,
    ):
        super().__init__(*args_trainer, **kwargs_trainer)
        self.custom_lr_params = custom_lr_params if custom_lr_params is not None else {}
        self.dataset_configs = dataset_configs if dataset_configs is not None else {}


    # override the get_multi_dataset_batch_sampler method
    def get_multi_dataset_batch_sampler(
        self,
        dataset: ConcatDataset,
        batch_samplers: list[BatchSampler],
        generator: torch.Generator | None = None,
        seed: int | None = 0,
    ) -> BatchSampler:
        """
        Returns the appropriate multi-dataset batch sampler based on the ``multi_dataset_batch_sampler`` argument
        in ``self.args``. This batch sampler class supports ``__len__`` and ``__iter__`` methods, and is used as the
        ``batch_sampler`` to create the :class:`torch.utils.data.DataLoader`.

        .. note::
            Override this method to provide a custom multi-dataset batch sampler.

        Args:
            dataset (ConcatDataset): The concatenation of all datasets.
            batch_samplers (List[BatchSampler]): List of batch samplers for each dataset in the concatenated dataset.
            generator (torch.Generator, optional): Optional random number generator for shuffling the indices.
            seed (int, optional): Optional seed for the random number generator
        """

        multi_batch_sampler_kwargs = {
            "batch_samplers": batch_samplers,
            "generator": generator,
            "seed": seed,
        }

        # If the multi-dataset batch sampler is a WeightedBatchSampler, initialize it
        if (
            inspect.isclass(self.args.multi_dataset_batch_sampler)
            and issubclass(
                self.args.multi_dataset_batch_sampler,
                MultiDatasetDefaultBatchSampler,
            )
            and hasattr(
                self.args.multi_dataset_batch_sampler,
                "__name__",
            )
            and self.args.multi_dataset_batch_sampler.__name__ == "WeightedBatchSampler"
        ):

            multi_batch_sampler_kwargs.update(
                {
                    "dataset_configs": self.dataset_configs,
                },
            )
            return WeightedBatchSampler(dataset=dataset, **multi_batch_sampler_kwargs)

        # If the multi-dataset batch sampler is a DefaultBatchSampler subclass, initialize it
        if inspect.isclass(self.args.multi_dataset_batch_sampler) and issubclass(
            self.args.multi_dataset_batch_sampler,
            MultiDatasetDefaultBatchSampler,
        ):
            return self.args.multi_dataset_batch_sampler(
                dataset,
                **multi_batch_sampler_kwargs,
            )

        if callable(self.args.multi_dataset_batch_sampler):
            return self.args.multi_dataset_batch_sampler(
                dataset,
                **multi_batch_sampler_kwargs,
            )

        # Otherwise, it's an MultiDatasetBatchSamplers instance and we use the samplers that match the enum values
        if (
            self.args.multi_dataset_batch_sampler
            == MultiDatasetBatchSamplers.ROUND_ROBIN
        ):
            return RoundRobinBatchSampler(dataset=dataset, **multi_batch_sampler_kwargs)

        if (
            self.args.multi_dataset_batch_sampler
            == MultiDatasetBatchSamplers.PROPORTIONAL
        ):
            return ProportionalBatchSampler(
                dataset=dataset,
                **multi_batch_sampler_kwargs,
            )
        

class BetaAnnealCallback(TrainerCallback):
    """Advances the annealed-tanh β after every optimizer step."""

    def __init__(self, layer):
        self.layer = layer

    def on_step_end(self, args, state, control, **kwargs):
        self.layer.anneal_step()
        if state.global_step % args.logging_steps == 0 and distributed.is_main_process():
            log_data = {"annealed_tanh/beta": self.layer.beta}
            if isinstance(self.layer, MultiBitAnnealedSigmoidLayer):
                t = self.layer.thresholds.detach()
                gaps = t[1:] - t[:-1]
                log_data.update({
                    "multibit_asigmoid/threshold_span": (t[-1] - t[0]).item(),
                    "multibit_asigmoid/min_gap": gaps.min().item() if len(gaps) > 0 else 0.0,
                    "multibit_asigmoid/anchor": t[0].item(),
                    **{f"multibit_asigmoid/t{k}": t[k].item() for k in range(len(t))},
                })
            wandb.log(log_data, step=state.global_step)


def initilize_model(local_rank):
    global config

    if config["experiment"]["number"] in ["e1", "e3"]:
        model = SentenceTransformer(
            config["input_model"]["name"],
            device=f"cuda:{local_rank}",
            tokenizer_kwargs={"model_max_length": config["input_model"]["max_len"], "truncation": True},
            model_kwargs={"torch_dtype": torch.bfloat16 if bf16_supported else None},
        )
    elif config["experiment"]["number"] in ["e2", "e4"]:
        word_embedding_model = models.Transformer(config["input_model"]["name"])
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
        )

        # Add our custom binarization layer after the pooling layer
        binarization_model = BinarizationLayer()

        # Create the final model by sequencing the layers
        model = SentenceTransformer(
            modules=[
                word_embedding_model,
                pooling_model,
                binarization_model,
            ],
            device=f"cuda:{local_rank}",
            tokenizer_kwargs={"model_max_length": config["input_model"]["max_len"], "truncation": True},
            model_kwargs={"torch_dtype": torch.bfloat16 if bf16_supported else None},
        )
    elif config["experiment"]["number"] == "e5":
        word_embedding_model = models.Transformer(config["input_model"]["name"])
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
        )
        binarization_model = AnnealedTanhBinarizationLayer(
            gamma=config["annealed_tanh_config"]["gamma"]
        )
        model = SentenceTransformer(
            modules=[
                word_embedding_model,
                pooling_model,
                binarization_model,
            ],
            device=f"cuda:{local_rank}",
            tokenizer_kwargs={"model_max_length": config["input_model"]["max_len"], "truncation": True},
            model_kwargs={"torch_dtype": torch.bfloat16 if bf16_supported else None},
        )
    elif config["experiment"]["number"] == "e7":
        word_embedding_model = models.Transformer(config["input_model"]["name"])
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
        )
        binarization_model = AnnealedTanhBinarizationLayer(
            gamma=config["annealed_tanh_config"]["gamma"]
        )
        model = SentenceTransformer(
            modules=[
                word_embedding_model,
                pooling_model,
                binarization_model,
            ],
            device=f"cuda:{local_rank}",
            tokenizer_kwargs={"model_max_length": config["input_model"]["max_len"], "truncation": True},
            model_kwargs={"torch_dtype": torch.bfloat16 if bf16_supported else None},
        )
    elif config["experiment"]["number"].startswith("e6_"):
        bits = int(config["experiment"]["number"].split("_")[1][:-1])  # e6_2b -> 2
        word_embedding_model = models.Transformer(config["input_model"]["name"])
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
        )
        quant_layer = MultiBitAnnealedSigmoidLayer(
            bits=bits,
            gamma=config["multibit_asigmoid_config"]["gamma"],
            init_scale=config["multibit_asigmoid_config"]["init_scale"],
            warmup_steps=config["multibit_asigmoid_config"]["warmup_steps"],
        )
        model = SentenceTransformer(
            modules=[
                word_embedding_model,
                pooling_model,
                quant_layer,
            ],
            device=f"cuda:{local_rank}",
            tokenizer_kwargs={"model_max_length": config["input_model"]["max_len"], "truncation": True},
            model_kwargs={"torch_dtype": torch.bfloat16 if bf16_supported else None},
        )

    return model


# ──────────────── Main ────────────────
def main(local_rank, rank):
    global args
    model = initilize_model(local_rank)
    ds_config = build_dataset_configs(config["input_data"]["n_data_src"])
    ds_dict = load_and_cache_datasets(ds_config, config["outputs"]["cache_dir"], config["input_data"]["nrows"], rank)


    collator = None
    train_ds = {n: s["train"] for n, s in ds_dict.items()}
    val_ds = {n: s["validation"] for n, s in ds_dict.items()}

    total_rows = (
        sum(len(d) for d in train_ds.values())
        + sum(len(d) for d in val_ds.values())
    )

    if distributed.is_main_process():
        config["train_data_points"] = sum(len(d) for d in train_ds.values())
        config["val_data_points"] = sum(len(d) for d in val_ds.values())
        config["total_datapoints"] = total_rows
        wandb.config.update(config, allow_val_change=True)

    if config["resume_config"]["resume_checkpoint_path"] is not None:
        output_dir = "/".join(config["resume_config"]["resume_checkpoint_path"].split("/")[:-2])
    else:
        output_dir = str(
            os.path.join(
                config["outputs"]["output_base"],
                f"nrows_{config['input_data']['nrows']}__nsrc_{config['input_data']['n_data_src']}",
                f"timestamp_{formatted_datetime}",
                config["input_model"]["name"].split("/")[-1],
            ),
        )
    os.makedirs(output_dir, exist_ok=True)

    print(f"RANK:{rank};Total rows: {total_rows}")

    if distributed.is_main_process():
        # eval_strategy = "steps"
        eval_cache_dir = config["outputs"]["cache_dir"] + "_evaluator/validation"
        val_evaluator = prepare_evaluators(
            {n: s["validation"] for n, s in ds_dict.items()},
            max_per_split=config["input_data"]["max_datapoints_per_src_for_eval"],
            BATCH_SIZE=config["trainer_config"]["per_device_eval_batch_size"],
            cache_dir=eval_cache_dir,
        )
    else:
        # eval_strategy = "no"
        val_evaluator = None
    

    args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        **config["trainer_config"],
    )

    # loss functions
    loss_sim_fun = util.cos_sim
    loss_funs = {
        n: cfg["loss"](model, similarity_fct=loss_sim_fun) for n, cfg in ds_config.items()
    }
    if config["experiment"]["number"] in ["e3", "e4", "e7"]:
        loss_funs = {
            n: MatryoshkaLoss(model=model, loss=base_loss, matryoshka_dims=config.get("mrl_config").get("matryoshka_dims") + [model.get_sentence_embedding_dimension()]) 
            for n, base_loss in loss_funs.items()
        }

    anneal_layer = next(
        (m for m in model.modules()
         if isinstance(m, (AnnealedTanhBinarizationLayer, MultiBitAnnealedSigmoidLayer))), None
    )
    callbacks = [BetaAnnealCallback(anneal_layer)] if anneal_layer is not None else []

    # On resume, the HF Trainer restores weights/optimizer/global_step but NOT
    # the annealing layer's plain-int `_step` (it is not in the state_dict). Without
    # this, beta would reset to 1 and re-climb from scratch instead of continuing
    # from where it left off (e.g. ~28 at step 8189). Restore `_step` from the
    # resumed checkpoint's global_step (== number of optimizer steps == anneal steps).
    if config["resume_config"]["resume_checkpoint_path"] is not None and anneal_layer is not None:
        with open(os.path.join(config["resume_config"]["resume_checkpoint_path"], "trainer_state.json")) as f:
            anneal_layer._step = json.load(f).get("global_step", 0)
        if distributed.is_main_process():
            print(f"[anneal] resumed _step={anneal_layer._step}, beta={anneal_layer.beta:.3f}")

    trainer = CustomSentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=None,
        loss=loss_funs,
        evaluator=val_evaluator,
        data_collator=collator,
        dataset_configs=ds_config,  # Pass dataset configs for WeightedBatchSampler
        callbacks=callbacks,
    )

    if config["resume_config"]["resume_checkpoint_path"] is not None:
        print(f"RANK:{rank};Resuming training...")
        trainer.train(resume_from_checkpoint=config["resume_config"]["resume_checkpoint_path"])
    else:
        print(f"RANK:{rank};Starting training...")
        trainer.train()

    print(f"RANK:{rank};Finished training...")

    torch.distributed.barrier(device_ids=[local_rank])

    if distributed.is_main_process(): 
        final_model_path = os.path.join(output_dir, "final_model")
        # Save the complete SentenceTransformer model, which includes all config files
        model.save_pretrained(final_model_path) 
        
        # --- NEW STEP: Log the saved directory as a WandB Artifact ---
        artifact = wandb.Artifact("final-sentence-transformer-model", type="model")
        artifact.add_dir(final_model_path)
        wandb.log_artifact(artifact)
        # -----------------------------------------------------------


if __name__ == "__main__":
    run = None
    local_rank, rank = init_ddp(True)
    # update the config with the rank and local_rank
    config["local_rank"] = local_rank
    config["trainer_config"]["eval_strategy"] = "steps" if rank == 0 else "no"
    
    print(f"RANK: {rank}; LOCAL_RANK: {local_rank}.")

    if distributed.is_main_process():
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        if config["resume_config"]["resume_run_id"] is not None:
            assert config["resume_config"]["resume_run_id"] is not None
            wandb.init(
                # entity=",
                project=config["logging"]["wb_project"],
                mode=config["logging"]["wb_mode"],
                id=config["resume_config"]["resume_run_id"],
                resume="must",
            )
        else:
            wandb.init(
                # entity="",
                project=config["logging"]["wb_project"],
                mode=config["logging"]["wb_mode"],
            )

        config = {**config, **get_gpu_info()}

    torch.distributed.barrier(device_ids=[local_rank])

    start_time = time.time()
    main(local_rank, rank)

    if distributed.is_main_process():
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"RANK: {rank}; Elapsed time: {elapsed_time:.2f} seconds")
        wandb.log({"timing/overall_script_seconds": elapsed_time})
        hours = elapsed_time / 3600
        wandb.log({"timing/overall_script_hours": hours})
        wandb.finish()

    torch.distributed.destroy_process_group()
