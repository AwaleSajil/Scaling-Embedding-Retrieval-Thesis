import argparse
import hashlib
import json
import random
import os
import pathlib
import time
from collections import defaultdict
from string import Template

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns
import torch
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader

from custum_evals import (
    DummyModel,
    MultiGPUInformationRetrievalEvaluator,
    MultiGPUNanoBEIREvaluator,
    UBinarySentenceTransformer,
    QuantizedSentenceTransformer,
    PQSentenceTransformer,
    TurboQuantSentenceTransformer,
    get_embedding_for_dataset,
    hamming_similarity_from_distance,
)
from datasets import load_dataset
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers import models as s_models

load_dotenv()
import sys

sys.path.append("../src/")  # Add training dir to path

parser = argparse.ArgumentParser(description="Sentence Transformer Training Config")

parser.add_argument(
    "--dataset_name",
    type=str,
    default="nanobeir",
    choices=[
        "nanobeir",
        "beir",
        "nasa_sde_ir_v3",
        "nasa_sde_ir_20251024_v5",
        "nasa_smd_ir",
    ],
)
parser.add_argument("--ks", nargs="*", default=[1, 3, 5, 10])
parser.add_argument("--plotks", nargs="*", default=[1, 3, 5, 10])
parser.add_argument("--json_output_path", type=str, default="results_json/")
parser.add_argument("--output_dir_plots", type=str, default="results_plots/")
parser.add_argument("--json_time_path", type=str, default="results_times/")
parser.add_argument("--emb_info_path", type=str, default="results_emb_info/")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument(
    "--just_plot",
    type=int,
    default=0,
    choices=[0, 1],
    help="Set to 1 to just plot the results without running the evaluation.",
)
parser.add_argument(
    "--desired_metric_types",
    nargs="+",
    default=["mrr"],
    help="A list of metrics to plot (e.g., mrr, accuracy, ndcg, precision, recall, map).",
)
parser.add_argument(
    "--embedding_cache_dir",
    type=str,
    default="embedding_cache/",
    help=(
        "Directory for caching raw corpus embeddings. "
        "Structure: <cache_dir>/<model_hash>/<dataset>/<subset>/<datafile>/corpus/. "
    ),
)
parser.add_argument(
    "--interactive_plot_path",
    type=str,
    default="result_interactive_plot/",
    help="Directory to write the self-contained interactive HTML dashboard.",
)


args = parser.parse_args()

dataset_name = args.dataset_name
ks = args.ks
plotks = [int(k) for k in args.plotks]
json_output_path = args.json_output_path
json_time_path = args.json_time_path
emb_info_dir = args.emb_info_path
output_dir_plots = args.output_dir_plots
batch_size = args.batch_size
interactive_plot_path = args.interactive_plot_path
just_plot = args.just_plot
desired_metric_types = args.desired_metric_types
embedding_cache_dir = args.embedding_cache_dir


os.makedirs(json_output_path, exist_ok=True)
os.makedirs(json_time_path, exist_ok=True)
os.makedirs(emb_info_dir, exist_ok=True)
os.makedirs(output_dir_plots, exist_ok=True)
json_time_path = f"{json_time_path}/{dataset_name}_corpus_embedding_times.json"
json_output_path = f"{json_output_path}/{dataset_name}_eval_dump.json"
emb_info_path = os.path.join(emb_info_dir, f"{dataset_name}_emb_info.json")


similarity_fns = {
    "hamming": hamming_similarity_from_distance,
}

models = {
    # --- Baselines (blue family, hatch: //) ---
    # model 0
    "bge-base-en-v1.5": {
        "path": "BAAI/bge-base-en-v1.5",
        "color": "#1f77b4",  # dark blue
        "hatch": "//",
        "model_config": {},
        "display_name": "BGE-base",
    },
    # model 1
    "finetuned-bge-base-en-v1.5": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        "color": "#6baed6",  # lighter blue
        "hatch": "//",
        "model_config": {},
        "display_name": "FT",
    },
    # --- Post-quantization from finetuned (purple family, hatch: xx) ---
    "finetuned-bge-base-en-v1.5-post-int8": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        "color": "#756bb1",  # medium purple
        "hatch": "xx",
        "quant_bits": 8,
        "display_name": "FT + INT8",
    },
    "finetuned-bge-base-en-v1.5-post-binary": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        "color": "#bcbddc",  # light purple
        "hatch": "xx",
        "similarity_fn_name": "hamming",
        "display_name": "FT + Bin",
    },
    # --- BAT (brown, hatch: --) ---
    "finetuned-bge-base-en-v1.5-bat": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_21-07-12/bge-base-en-v1.5/final_model",
        "color": "#8c564b",  # brown
        "hatch": "--",
        "similarity_fn_name": "hamming",
        "display_name": "BAT",
    },
    # --- MRL float (green gradient, lighter=smaller dim, hatch: ||) ---
    "finetuned-bge-base-en-v1.5-mrl-32": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        "color": "#c7e9c0",  # lightest green
        "hatch": "||",
        "truncate_dim": 32,
        "display_name": "MRL-32",
    },
    "finetuned-bge-base-en-v1.5-mrl-64": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        "color": "#74c476",  # light green
        "hatch": "||",
        "truncate_dim": 64,
        "display_name": "MRL-64",
    },
    "finetuned-bge-base-en-v1.5-mrl-128": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        "color": "#31a354",  # medium green
        "hatch": "||",
        "truncate_dim": 128,
        "display_name": "MRL-128",
    },
    "finetuned-bge-base-en-v1.5-mrl-256": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        "color": "#006d2c",  # dark green
        "hatch": "||",
        "truncate_dim": 256,
        "display_name": "MRL-256",
    },
    "finetuned-bge-base-en-v1.5-mrl-512": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        "color": "#00441b",  # darkest green
        "hatch": "||",
        "truncate_dim": 512,
        "display_name": "MRL-512",
    },
    # --- MRL + post-binary (orange→red gradient, lighter=smaller dim, hatch: \\) ---
    "finetuned-bge-base-en-v1.5-mrl-32-post-binary": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        "color": "#fdd0a2",  # lightest orange
        "hatch": "\\\\",
        "truncate_dim": 32,
        "similarity_fn_name": "hamming",
        "display_name": "MRL-32 + Bin",
    },
    "finetuned-bge-base-en-v1.5-mrl-64-post-binary": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        "color": "#fdae6b",  # light orange
        "hatch": "\\\\",
        "truncate_dim": 64,
        "similarity_fn_name": "hamming",
        "display_name": "MRL-64 + Bin",
    },
    "finetuned-bge-base-en-v1.5-mrl-128-post-binary": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        "color": "#f16913",  # medium orange
        "hatch": "\\\\",
        "truncate_dim": 128,
        "similarity_fn_name": "hamming",
        "display_name": "MRL-128 + Bin",
    },
    "finetuned-bge-base-en-v1.5-mrl-256-post-binary": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        "color": "#d94801",  # dark orange-red
        "hatch": "\\\\",
        "truncate_dim": 256,
        "similarity_fn_name": "hamming",
        "display_name": "MRL-256 + Bin",
    },
    "finetuned-bge-base-en-v1.5-mrl-512-post-binary": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        "color": "#7f2704",  # darkest red
        "hatch": "\\\\",
        "truncate_dim": 512,
        "similarity_fn_name": "hamming",
        "display_name": "MRL-512 + Bin",
    },
    # --- PQ only, full dim=768 (rose/crimson family, darker=more bits, hatch: ++) ---
    "finetuned-bge-base-en-v1.5-pq-m16-n8": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        "color": "#c9184a",  # dark crimson  — 128 bits/vec
        "hatch": "++",
        "pq_M": 16,       # 768 / 16 = 48 dims per subspace
        "pq_nbits": 8,
        "display_name": "PQ (M16, b8)",
    },
    "finetuned-bge-base-en-v1.5-pq-m16-n4": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        "color": "#ff758f",  # light rose    — 64 bits/vec
        "hatch": "++",
        "pq_M": 16,       # 768 / 16 = 48 dims per subspace
        "pq_nbits": 4,
        "display_name": "PQ (M16, b4)",
    },
    # --- MRL + PQ (navy→teal gradient, darker=larger dim/more bits, hatch: oo) ---
    "finetuned-bge-base-en-v1.5-mrl-512-pq-m16-n8": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        "color": "#03045e",  # deepest navy   — 512 dims × 8 bits = 4096 bits/vec
        "hatch": "oo",
        "truncate_dim": 512,
        "pq_M": 16,       # 512 / 16 = 32 dims per subspace
        "pq_nbits": 8,
        "display_name": "MRL-512 + PQ (M16, b8)",
    },
    "finetuned-bge-base-en-v1.5-mrl-256-pq-m16-n8": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        "color": "#023e8a",  # very dark blue — 256 dims × 8 bits = 2048 bits/vec
        "hatch": "oo",
        "truncate_dim": 256,
        "pq_M": 16,       # 256 / 16 = 16 dims per subspace
        "pq_nbits": 8,
        "display_name": "MRL-256 + PQ (M16, b8)",
    },
    "finetuned-bge-base-en-v1.5-mrl-128-pq-m16-n8": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        "color": "#014f86",  # dark navy     — 128 bits/vec
        "hatch": "oo",
        "truncate_dim": 128,
        "pq_M": 16,       # 128 / 16 = 8 dims per subspace
        "pq_nbits": 8,
        "display_name": "MRL-128 + PQ (M16, b8)",
    },
    "finetuned-bge-base-en-v1.5-mrl-128-pq-m16-n4": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        "color": "#0077b6",  # dark blue     — 64 bits/vec
        "hatch": "oo",
        "truncate_dim": 128,
        "pq_M": 16,       # 128 / 16 = 8 dims per subspace
        "pq_nbits": 4,
        "display_name": "MRL-128 + PQ (M16, b4)",
    },
    "finetuned-bge-base-en-v1.5-mrl-64-pq-m8-n8": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        "color": "#0096c7",  # medium teal   — 64 bits/vec
        "hatch": "oo",
        "truncate_dim": 64,
        "pq_M": 8,        # 64 / 8 = 8 dims per subspace
        "pq_nbits": 8,
        "display_name": "MRL-64 + PQ (M8, b8)",
    },
    "finetuned-bge-base-en-v1.5-mrl-64-pq-m8-n4": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        "color": "#00b4d8",  # light teal    — 32 bits/vec
        "hatch": "oo",
        "truncate_dim": 64,
        "pq_M": 8,        # 64 / 8 = 8 dims per subspace
        "pq_nbits": 4,
        "display_name": "MRL-64 + PQ (M8, b4)",
    },
    "finetuned-bge-base-en-v1.5-mrl-32-pq-m4-n8": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        "color": "#48cae4",  # sky blue      — 32 bits/vec
        "hatch": "oo",
        "truncate_dim": 32,
        "pq_M": 4,        # 32 / 4 = 8 dims per subspace
        "pq_nbits": 8,
        "display_name": "MRL-32 + PQ (M4, b8)",
    },
    "finetuned-bge-base-en-v1.5-mrl-32-pq-m4-n4": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        "color": "#90e0ef",  # pale sky      — 16 bits/vec
        "hatch": "oo",
        "truncate_dim": 32,
        "pq_M": 4,        # 32 / 4 = 8 dims per subspace
        "pq_nbits": 4,
        "display_name": "MRL-32 + PQ (M4, b4)",
    },
    # --- MRL + BAT (pink/magenta gradient, lighter=smaller dim, hatch: **) ---
    # All share the MRL+Binarization-layer checkpoint (experiment 4)
    "finetuned-bge-base-en-v1.5-mrl-notrunc-bat": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260411_10-35-45/bge-base-en-v1.5/final_model",
        "color": "#7b0051",  # darkest magenta — 768 bits/vec
        "hatch": "**",
        "similarity_fn_name": "hamming",
        "display_name": "MRL-BAT",
    },
    "finetuned-bge-base-en-v1.5-mrl-512-bat": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260411_10-35-45/bge-base-en-v1.5/final_model",
        "color": "#ad1457",  # dark pink       — 512 bits/vec
        "hatch": "**",
        "truncate_dim": 512,
        "similarity_fn_name": "hamming",
        "display_name": "MRL-512+BAT",
    },
    "finetuned-bge-base-en-v1.5-mrl-256-bat": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260411_10-35-45/bge-base-en-v1.5/final_model",
        "color": "#e91e8c",  # medium magenta  — 256 bits/vec
        "hatch": "**",
        "truncate_dim": 256,
        "similarity_fn_name": "hamming",
        "display_name": "MRL-256+BAT",
    },
    "finetuned-bge-base-en-v1.5-mrl-128-bat": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260411_10-35-45/bge-base-en-v1.5/final_model",
        "color": "#f06292",  # light pink      — 128 bits/vec
        "hatch": "**",
        "truncate_dim": 128,
        "similarity_fn_name": "hamming",
        "display_name": "MRL-128+BAT",
    },
    "finetuned-bge-base-en-v1.5-mrl-64-bat": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260411_10-35-45/bge-base-en-v1.5/final_model",
        "color": "#f8bbd0",  # lighter pink    — 64 bits/vec
        "hatch": "**",
        "truncate_dim": 64,
        "similarity_fn_name": "hamming",
        "display_name": "MRL-64+BAT",
    },
    "finetuned-bge-base-en-v1.5-mrl-32-bat": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260411_10-35-45/bge-base-en-v1.5/final_model",
        "color": "#fce4ec",  # lightest pink   — 32 bits/vec
        "hatch": "**",
        "truncate_dim": 32,
        "similarity_fn_name": "hamming",
        "display_name": "MRL-32+BAT",
    },
    # --- TurboQuant post-hoc (gold/amber gradient, darker=more bits, hatch: OO) ---
    # Applied to the standard fine-tuned checkpoint (experiment 1)
    "finetuned-bge-base-en-v1.5-tq-8bit": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        "color": "#7b4f00",  # very dark gold — 8 bits/dim → 6144 bits/vec
        "hatch": "OO",
        "tq_bits": 8,
        "display_name": "FT + TQ-8b",
    },
    "finetuned-bge-base-en-v1.5-tq-4bit": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        "color": "#d4a017",  # dark gold     — 4 bits/dim → 3072 bits/vec
        "hatch": "OO",
        "tq_bits": 4,
        "display_name": "FT + TQ-4b",
    },
    "finetuned-bge-base-en-v1.5-tq-3bit": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        "color": "#f0c040",  # medium gold   — 3 bits/dim → 2304 bits/vec
        "hatch": "OO",
        "tq_bits": 3,
        "display_name": "FT + TQ-3b",
    },
    "finetuned-bge-base-en-v1.5-tq-2bit": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        "color": "#fce48a",  # light gold    — 2 bits/dim → 1536 bits/vec
        "hatch": "OO",
        "tq_bits": 2,
        "display_name": "FT + TQ-2b",
    },
    "finetuned-bge-base-en-v1.5-tq-1bit": {
        "path": "/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        "color": "#fff5cc",  # very light gold — 1 bit/dim → 768 bits/vec
        "hatch": "OO",
        "tq_bits": 1,
        "display_name": "FT + TQ-1b",
    },

    }

dataset_config = {
    "nanobeir": {
        "path": None,
        "subsets": [None],  # this means to use all datasets
        "paths": {
            "NanoClimateFEVER": "zeta-alpha-ai/NanoClimateFEVER",
            "NanoDBPedia": "zeta-alpha-ai/NanoDBPedia",
            "NanoFEVER": "zeta-alpha-ai/NanoFEVER",
            "NanoFiQA2018": "zeta-alpha-ai/NanoFiQA2018",
            "NanoHotpotQA": "zeta-alpha-ai/NanoHotpotQA",
            "NanoMSMARCO": "zeta-alpha-ai/NanoMSMARCO",
            "NanoNFCorpus": "zeta-alpha-ai/NanoNFCorpus",
            "NanoNQ": "zeta-alpha-ai/NanoNQ",
            "NanoQuoraRetrieval": "zeta-alpha-ai/NanoQuoraRetrieval",
            "NanoSCIDOCS": "zeta-alpha-ai/NanoSCIDOCS",  # issue
            "NanoArguAna": "zeta-alpha-ai/NanoArguAna",
            "NanoSciFact": "zeta-alpha-ai/NanoSciFact",
            "NanoTouche2020": "zeta-alpha-ai/NanoTouche2020",
        },
        # “climatefever”, “dbpedia”, “fever”, “fiqa2018”, “hotpotqa”, “msmarco”, 
        # “nfcorpus”, “nq”, “quoraretrieval”, “scidocs”, “arguana”, 
        # “scifact”, and “touche2020”.
        # "dataset_names": ["climatefever"] # if None all the subsets will be used
    },
    "beir": {
        "path": None,
        "subsets": [
            "trec-covid",
            "nfcorpus",
            "nq",
            "hotpotqa",
            "fiqa",
            "arguana",
            "webis-touche2020",
            "dbpedia-entity",
            "scidocs",
            "fever",
            "climate-fever",
            "scifact",
        ],
        "dataset_cache_path": "./beir_datasets",
    },
    "nasa_smd_ir": {"path": "nasa-impact/nasa-smd-IR-benchmark"},
    "nasa_sde_ir_v3": {
        "path": "nasa-impact/nasa-sde-IR-benchmark-sample-v3",
        "data_files": [
            "qrels/question-answer~SDE_general_v2.tsv",
            "qrels/question-answer~SDE_general_v3.tsv",
            "qrels/search_term-document~CMR.tsv",
            "qrels/search_term-document~PDS.tsv",
            "qrels/search_term-document~SDE_general_v2.tsv",
            "qrels/search_term-document~SDE_general_v3.tsv",
            "qrels/title-description~CMR.tsv",
            "qrels/title-description~PDS.tsv",
        ],
        "data_files_colors": [
            "#1f77b4",  # question-answer~SDE_general_v2.tsv
            "#ff7f0e",  # question-answer~SDE_general_v3.tsv
            "#2ca02c",  # search_term-document~CMR.tsv
            "#d62728",  # search_term-document~PDS.tsv
            "#9467bd",  # search_term-document~SDE_general_v2.tsv
            "#8c564b",  # search_term-document~SDE_general_v3.tsv
            "#e377c2",  # title-description~CMR.tsv
            "#7f7f7f",  # title-description~PDS.tsv
        ],
    },
    "nasa_sde_ir_20251024_v5": {
        "path": "nasa-impact/nasa-sde-IR-benchmark-20251024-v5",
        "data_files": [
            "qrels/qa_pairs.tsv",
            "qrels/search_pairs.tsv",
        ],
        "data_files_colors": [
            "#1f77b4",  # qa_pairs.tsv
            "#ff7f0e",  # search_pairs.tsv
        ]
    }
}


def dataset_getter(
    dataset_name,
    corpus_split="train",
    queries_split="train",
    relevant_docs_split="test",
    data_file=None,
):
    corpus = load_dataset(
        dataset_config[dataset_name]["path"],
        data_files="corpus.jsonl",
        split=corpus_split,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )
    queries = load_dataset(
        dataset_config[dataset_name]["path"],
        data_files="queries.jsonl",
        split=queries_split,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )
    relevant_docs_data = load_dataset(
        dataset_config[dataset_name]["path"],
        split=relevant_docs_split,
        data_files=data_file,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )

    corpus = {row["_id"]: row["text"] for i, row in enumerate(corpus)}
    queries = {row["_id"]: row["text"] for row in queries}
    relevant_docs_data = (
        relevant_docs_data.to_pandas()
        .groupby("query-id")["corpus-id"]
        .apply(set)
        .to_dict()
    )
    relevant_docs_data = {
        str(k): {str(item) for item in v} for k, v in relevant_docs_data.items()
    }

    return corpus, queries, relevant_docs_data


def beir_dataset_getter(subset):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{subset}.zip"
    out_dir = os.path.join(
        pathlib.Path.cwd(),
        dataset_config["beir"]["dataset_cache_path"],
    )
    data_path = beir_util.download_and_unzip(url, out_dir)

    # Load the dataset using the BEIR loader
    corpus_beir, queries, qrels_beir = GenericDataLoader(data_folder=data_path).load(
        split="test",
    )

    # The corpus needs to be flattened from Dict[str, Dict[str,str]] to Dict[str, str]
    # We'll concatenate the title and text for each document.
    corpus = {
        doc_id: (doc.get("title", "") + " " + doc.get("text", "")).strip()
        for doc_id, doc in corpus_beir.items()
    }

    # The qrels need to be converted to the relevant_docs format: Dict[str, Set[str]]
    # We'll only consider documents with a relevance score > 0.
    relevant_docs = defaultdict(set)
    for query_id, doc_scores in qrels_beir.items():
        for doc_id, score in doc_scores.items():
            if score > 0:
                relevant_docs[query_id].add(doc_id)

    return corpus, queries, relevant_docs


def get_dataset(dataset_name, subset=None, relevant_docs_split="test", data_file=None):
    if dataset_name.lower() in ["beir"]:
        return beir_dataset_getter(subset)
    elif dataset_name.lower() in ["nanobeir"]:
        return None, None, None
    else:
        return dataset_getter(
            dataset_name,
            relevant_docs_split=relevant_docs_split,
            data_file=data_file,
        )


def get_evaluator(
    dataset_name: str,
    queries: dict,
    corpus: dict,
    relevant_docs_data: dict,
    subset=None,
    data_file=None,
    dataset_config=None
):

    if dataset_name.lower() == "nanobeir":
        args = dict(
            dataset_names=dataset_config.get("dataset_names", None),
            mrr_at_k=ks,
            accuracy_at_k=ks,
            precision_recall_at_k=ks,
            map_at_k=ks,
            ndcg_at_k=ks,
            show_progress_bar=True,
            batch_size=batch_size,
            write_csv=True,
        )
        evaluators = {
            **{"cosine": MultiGPUNanoBEIREvaluator(**args)},
            **{
                name: MultiGPUNanoBEIREvaluator(**args, score_functions={name: fn})
                for name, fn in similarity_fns.items()
            },
        }

    elif dataset_name.lower() == "beir":
        args = dict(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs_data,
            name=f"beir__{subset}____evaluator",
            batch_size=batch_size,
            mrr_at_k=ks,
            ndcg_at_k=ks,
            accuracy_at_k=ks,
            precision_recall_at_k=ks,
            map_at_k=ks,
            show_progress_bar=True,
            write_csv=True,
            encode_chunk_size=5000,
            encode_batch_size=batch_size,
        )
        evaluators = {
            **{"cosine": MultiGPUInformationRetrievalEvaluator(**args)},
            **{
                name: MultiGPUInformationRetrievalEvaluator(
                    **args,
                    score_functions={name: fn},
                )
                for name, fn in similarity_fns.items()
            },
        }

    elif dataset_name.lower() in [
        "nasa_sde_ir_v3",
        "nasa_sde_ir_20251024_v5",
        "nasa_repo_code_benchmark_v0.2",
        "nasa_repo_code_benchmark_v0.3",
    ]:
        args = dict(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs_data,
            name=f"{dataset_name}____{data_file}__evaluator",
            batch_size=batch_size,
            mrr_at_k=ks,
            ndcg_at_k=ks,
            accuracy_at_k=ks,
            precision_recall_at_k=ks,
            map_at_k=ks,
            show_progress_bar=True,
            write_csv=True,
            encode_chunk_size=1000,
            encode_batch_size=batch_size,
            corpus_chunk_size=500,
        )
        evaluators = {
            **{"cosine": MultiGPUInformationRetrievalEvaluator(**args)},
            **{
                name: MultiGPUInformationRetrievalEvaluator(
                    **args,
                    score_functions={name: fn},
                )
                for name, fn in similarity_fns.items()
            },
        }

    elif dataset_name.lower() == "nasa_smd_ir":
        args = dict(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs_data,
            name=f"{dataset_name}______evaluator",
            batch_size=batch_size,
            mrr_at_k=ks,
            ndcg_at_k=ks,
            accuracy_at_k=ks,
            precision_recall_at_k=ks,
            map_at_k=ks,
            show_progress_bar=True,
            write_csv=True,
            encode_chunk_size=5000,
            encode_batch_size=batch_size,
        )
        evaluators = {
            **{"cosine": MultiGPUInformationRetrievalEvaluator(**args)},
            **{
                name: MultiGPUInformationRetrievalEvaluator(
                    **args,
                    score_functions={name: fn},
                )
                for name, fn in similarity_fns.items()
            },
        }

    return evaluators


def add_mean_metrics(all_results, query_counts, mean_basis="subset"):
    global models
    for model_name in all_results:

        similarity_fn_names = set(
            [i.split("_")[-2] for i in all_results[model_name].keys()],
        )
        if len(similarity_fn_names) > 1:
            raise ValueError(
                f"Multiple similarity functions found for {model_name}: "
                f"{similarity_fn_names}. Please handle manually. Skipping...",
            )

        similarity_fn_name = similarity_fn_names.pop()
        # similarity_fn_name = models.get(model_name, {}).get("similarity_fn_name", "cosine")
        metric_names = set(
            [i.split("_")[-1] for i in list(all_results[model_name].keys())],
        )

        if mean_basis == "subset":
            mean_basis_names = set(
                [
                    i.split("__")[1]
                    for i in all_results[model_name]
                    if "mean" not in i.split("__")[1]
                ],
            )
            key_name = "${dataset_name}__${mean_basis_name}____evaluator_${similarity_fn_name}_${metric}"
            result_key_name = "${dataset_name}__${mean_type}____evaluator_${similarity_fn_name}_${metric}"
        elif mean_basis == "data_file":
            mean_basis_names = set(
                [
                    i.split("__")[2]
                    for i in all_results[model_name]
                    if "mean" not in i.split("__")[2]
                ],
            )
            key_name = "${dataset_name}____${mean_basis_name}__evaluator_${similarity_fn_name}_${metric}"
            result_key_name = "${dataset_name}____${mean_type}__evaluator_${similarity_fn_name}_${metric}"

        mean_result = {}
        weighted_mean_result = {}

        # Calculate the mean for each metric
        for metric in metric_names:
            values = []
            weighted_values = []
            weights = []
            for mean_basis_name in mean_basis_names:
                _key_name = Template(key_name).substitute(
                    dataset_name=dataset_name,
                    mean_basis_name=mean_basis_name,
                    similarity_fn_name=similarity_fn_name,
                    metric=metric,
                )
                try:
                    values.append(all_results[model_name][_key_name])
                except KeyError:
                    print(
                        f"Key {_key_name} not found in results for model {model_name}. "
                        f"Found only {list(all_results[model_name].keys())}. Skipping...",
                    )
                    continue
                weight_key = "__".join(_key_name.split("__")[:3])
                weight = query_counts.get(weight_key, 1)
                weighted_values.append(values[-1] * weight)
                weights.append(weight)

            _result_key_name = Template(result_key_name).substitute(
                dataset_name=dataset_name,
                mean_type="mean",
                similarity_fn_name=similarity_fn_name,
                metric=metric,
            )
            mean_result[_result_key_name] = sum(
                values,
            ) / (len(values) if len(values) > 0 else 1)

            if sum(weights) > 0:
                __result_key_name = Template(result_key_name).substitute(
                    dataset_name=dataset_name,
                    mean_type="weightedmean",
                    similarity_fn_name=similarity_fn_name,
                    metric=metric,
                )
                weighted_mean_result[__result_key_name] = sum(
                    weighted_values,
                ) / sum(weights)
            else:
                weighted_mean_result[__result_key_name] = 0.0

        all_results[model_name] = {
            **all_results[model_name],
            **mean_result,
            **weighted_mean_result,
        }


def check_if_eval_already_exists(all_results, model_name, subset=None, data_file=None):
    if model_name not in all_results:
        return False

    # get set of all data_files for the model_name
    existing_data_files = set(
        [k.split("__")[2] for k in all_results.get(model_name, {}).keys()],
    )
    if data_file is not None and data_file not in existing_data_files:
        return False

    # similarly get all the subsets for the model_name
    existing_subsets = set(
        [k.split("__")[1] for k in all_results.get(model_name, {}).keys()],
    )
    if subset is not None and subset not in existing_subsets:
        return False

    return True


def get_cache_path(model_path, dataset_name, subset, data_file, cache_dir):
    """Return the directory path for a cached corpus embedding."""
    model_hash = hashlib.sha256(model_path.encode()).hexdigest()[:16]
    subset_str = subset if subset is not None else "no_subset"
    if data_file is not None:
        # Use just the filename portion so paths like "qrels/foo.tsv" become "foo.tsv"
        data_file_str = os.path.basename(data_file)
    else:
        data_file_str = "no_datafile"
    return os.path.join(cache_dir, model_hash, dataset_name, subset_str, data_file_str, "corpus")


def get_or_compute_raw_corpus_embeddings(
    model_path, corpus_texts, cache_dir, dataset_name, subset, data_file, batch_size
):
    """
    Return raw float32 full-dim corpus embeddings, using disk cache when available.

    The cache stores a plain SentenceTransformer encoding with no truncation,
    binarization, or quantization applied.  All such transformations are applied
    later by apply_model_transformations().
    """
    cache_path = get_cache_path(model_path, dataset_name, subset, data_file, cache_dir)
    emb_file = os.path.join(cache_path, "embeddings.npy")
    meta_file = os.path.join(cache_path, "meta.json")

    if os.path.exists(emb_file):
        print(f"[EmbCache] Cache hit — loading from {emb_file}")
        return np.load(emb_file)

    print(f"[EmbCache] Cache miss — computing raw embeddings for model={model_path}, "
          f"dataset={dataset_name}, subset={subset}, data_file={data_file}")

    base_model = SentenceTransformer(model_path)
    pool = base_model.start_multi_process_pool()
    embeddings = base_model.encode(
        corpus_texts,
        pool=pool,
        batch_size=batch_size,
        chunk_size=1000,
        show_progress_bar=True,
    )
    base_model.stop_multi_process_pool(pool)

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    embeddings = np.asarray(embeddings, dtype=np.float32)

    os.makedirs(cache_path, exist_ok=True)
    np.save(emb_file, embeddings)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": model_path,
                "dataset_name": dataset_name,
                "subset": str(subset),
                "data_file": str(data_file),
                "n_texts": len(corpus_texts),
                "embedding_dim": int(embeddings.shape[1]),
            },
            f,
            indent=2,
        )
    print(f"[EmbCache] Saved to {emb_file}  shape={embeddings.shape}")
    return embeddings


def apply_model_transformations(raw_embeddings, model_info):
    """
    Apply model-specific post-processing to raw float32 full-dim corpus embeddings.

    Order of operations (matches what each model wrapper does during encode):
      1. Truncate dimensions (truncate_dim)
      2. Binarize  (similarity_fn_name == "hamming"  → UBinary)
      3. INT8/4 quantize (quant_bits)
      4. PQ / standard float: pass through as torch tensor

    For INT8/4 calibration we sample from the (truncated) corpus embeddings
    themselves — same distribution, avoids re-loading the model.
    """
    from sentence_transformers.quantization import quantize_embeddings

    embs = raw_embeddings  # float32 numpy, shape (N, full_dim)

    # 1. Truncate
    truncate_dim = model_info.get("truncate_dim", None)
    if truncate_dim is not None:
        embs = embs[:, :truncate_dim]

    similarity_fn = model_info.get("similarity_fn_name", "cosine")
    quant_bits = model_info.get("quant_bits", None)

    # 2. Binarize (UBinary / hamming)
    if similarity_fn == "hamming":
        embs = quantize_embeddings(embs, precision="ubinary")
        if not isinstance(embs, torch.Tensor):
            embs = torch.from_numpy(np.asarray(embs))

    # 3. INT8 / INT4 quantization
    elif quant_bits is not None:
        precision = {8: "int8", 4: "int4"}.get(quant_bits, "int8")
        n_calib = min(1000, len(embs))
        calib_embs = embs[:n_calib]
        embs = quantize_embeddings(embs, precision=precision, calibration_embeddings=calib_embs)
        if not isinstance(embs, torch.Tensor):
            embs = torch.from_numpy(np.asarray(embs))

    # 4. PQ or standard float — evaluator handles PQ fitting internally
    else:
        embs = torch.from_numpy(embs.copy())

    return embs


def pre_compute_corpus_embedding(
    models,
    dataset_name,
    subset,
    all_results,
    dataset_config,
    time_taken,
    calibration_texts,
    embedding_cache_dir=None,
):
    if dataset_name.lower() in ["nanobeir"]:
        print(
            f"Skipping pre-computation of corpus embeddings for {dataset_name} as it is not supported.",
        )
        return {}, {}
    data_file = dataset_config[dataset_name].get("data_files", [None])

    relevant_docs_split = "train" if any([i is not None for i in data_file]) else "test"
    corpus, _q, _ = get_dataset(
        dataset_name,
        subset,
        relevant_docs_split=relevant_docs_split,
        data_file=data_file[0] if data_file[0] is not None else None,
    )

    corpus_texts = list(corpus.values())
    corpus_pre_computed_embeddings = {}

    # --- Group models by their path so we only encode once per unique checkpoint ---
    path_to_model_names = {}
    for model_name, model_info in models.items():
        if check_if_eval_already_exists(all_results, model_name, subset, data_file[0]):
            print(
                f"Model {model_name} with the subset {subset} and data_file {data_file[0]} "
                f"already evaluated. Skipping...Preembedding of corpus",
            )
            continue
        model_path = model_info["path"]
        path_to_model_names.setdefault(model_path, []).append(model_name)

    for model_path, model_names in path_to_model_names.items():
        print(
            f"Pre-computing corpus embeddings for model path: {model_path}, "
            f"subset: {subset}  (shared by: {model_names})",
        )

        if embedding_cache_dir is not None:
            # --- Cache path: load raw float32 embeddings from disk or compute once ---
            start_time = time.time()
            raw_embeddings = get_or_compute_raw_corpus_embeddings(
                model_path=model_path,
                corpus_texts=corpus_texts,
                cache_dir=embedding_cache_dir,
                dataset_name=dataset_name,
                subset=subset,
                data_file=data_file[0],
                batch_size=batch_size,
            )
            encode_time = time.time() - start_time

            for model_name in model_names:
                model_info = models[model_name]
                if model_name not in time_taken:
                    time_taken[model_name] = {}
                time_taken[model_name][f"{dataset_name}__{subset}"] = encode_time

                # Apply truncation / binarization / quantization post-load
                corpus_pre_computed_embeddings[
                    f"{dataset_name}__{subset}__{model_name}"
                ] = apply_model_transformations(raw_embeddings, model_info)

        else:
            # --- No cache: original behaviour, load each model wrapper and encode ---
            for model_name in model_names:
                model_info = models[model_name]
                if model_name not in time_taken:
                    time_taken[model_name] = {}
                model = load_model_with_proper_pooling(
                    model_name, model_info, calibration_texts=calibration_texts
                )
                start_time = time.time()
                pool = model.start_multi_process_pool()
                corpus_embeddings = model.encode(
                    corpus_texts,
                    pool=pool,
                    batch_size=batch_size,
                    chunk_size=1000,
                    show_progress_bar=True,
                )
                model.stop_multi_process_pool(pool)
                time_taken[model_name][f"{dataset_name}__{subset}"] = time.time() - start_time
                corpus_pre_computed_embeddings[
                    f"{dataset_name}__{subset}__{model_name}"
                ] = corpus_embeddings

    return corpus_pre_computed_embeddings, time_taken


def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def generate_query_counts_for_nanobeir(dataset_config, dataset_name):
    query_counts = {}

    for d_name, path in dataset_config[dataset_name]["paths"].items():
        n_queries = load_dataset(
            path,
            split="train",
            name="queries",
            token=os.environ["HUGGINGFACE_TOKEN"],
        )
        query_counts[f"{dataset_name}__{d_name}__"] = len(n_queries)

    return query_counts


def get_calibration_corpus_texts(n_samples=1000):
    """
    Returns a list of raw texts to be used for calibration.
    Prioritizes the current dataset's corpus, falls back to NanoClimateFEVER.
    """
    import random
    from datasets import load_dataset

    # Access global variables
    config = dataset_config.get(dataset_name, {})
    
    texts = []
    source_log = ""

    try:
        # 1. Try: specific "path" defined in config (e.g. NASA datasets)
        if config.get("path"):
            source_log = f"path: {config.get('path')}"
            
            # Helper to handle local files vs HuggingFace hub
            data_files = config.get("data_files") # Check if we need specific files
            
            if data_files:
                # If specific files are defined (like in your NASA config), load the corpus file
                # usually "corpus.jsonl" is standard in your config logic
                dataset = load_dataset(
                    config.get("path"), 
                    data_files="corpus.jsonl", 
                    split="train"
                )
                texts = dataset["text"]
            else:
                # Standard HuggingFace load (try 'corpus' subset first)
                try:
                    dataset = load_dataset(config.get("path"), "corpus", split="train")
                    texts = dataset["text"]
                except:
                    # Fallback if 'corpus' subset config doesn't exist
                    dataset = load_dataset(config.get("path"), split="train")
                    texts = dataset["text"]

        # 2. Try: one of the "paths" (e.g. NanoBEIR style)
        elif config.get("paths"):
            # paths is a dict, so we take the first value available
            first_path = next(iter(config.get("paths").values()))
            source_log = f"paths (first): {first_path}"
            
            dataset = load_dataset(first_path, "corpus", split="train")
            texts = dataset["text"]

        # 3. Fallback: General Corpus
        else:
            source_log = "Fallback (NanoClimateFEVER)"
            dataset = load_dataset("zeta-alpha-ai/NanoClimateFEVER", "corpus", split="train")
            texts = dataset["text"]

        print(f"Loaded calibration data from {source_log}")

        # Sampling
        texts = list(texts)
        if len(texts) > n_samples:
            texts = random.sample(texts, n_samples)
        
        return texts

    except Exception as e:
        print(f"Warning: Could not load calibration data from {source_log} ({e}).")
        print("Using dummy texts for calibration.")
        return ["The quick brown fox jumps over the lazy dog."] * 100
    

def load_model_with_proper_pooling(model_name, model_info, calibration_texts=None):
    """
    Loads a model with specific pooling configurations, quantization wrappers, 
    or binary wrappers based on the model_info dictionary.
    """
    
    quant_bits = model_info.get("quant_bits", None)
    pq_M = model_info.get("pq_M", None)
    pq_nbits = model_info.get("pq_nbits", 8)
    tq_bits = model_info.get("tq_bits", None)
    similarity_fn = model_info.get("similarity_fn_name", "cosine")
    model_path = model_info["path"]
    truncate_dim = model_info.get("truncate_dim", None)
    model_kwargs = model_info.get("model_config", {})

    # Helper function to instantiate the correct class
    def _instantiate_model(modules=None, path=None, kwargs=None):
        # 1. Hamming/Binary Models (Highest Priority)
        if similarity_fn == "hamming":
            print(f"Using UBinarySentenceTransformer for {model_name}")
            if modules:
                return UBinarySentenceTransformer(modules=modules, truncate_dim=truncate_dim)
            return UBinarySentenceTransformer(path, truncate_dim=truncate_dim, model_kwargs=kwargs)

        # 2. Quantized Models (Int8/Int4)
        elif quant_bits is not None:
            print(f"Loading {model_name} as QuantizedSentenceTransformer ({quant_bits} bits)")
            if modules:
                return QuantizedSentenceTransformer(truncate_dim=truncate_dim, modules=modules, quant_bits=quant_bits, calibration_texts=calibration_texts)
            return QuantizedSentenceTransformer(path, truncate_dim=truncate_dim, quant_bits=quant_bits, calibration_texts=calibration_texts, model_kwargs=kwargs)

        # 3. Product Quantization Models
        elif pq_M is not None:
            print(f"Loading {model_name} as PQSentenceTransformer (M={pq_M}, nbits={pq_nbits})")
            if modules:
                return PQSentenceTransformer(truncate_dim=truncate_dim, modules=modules, M=pq_M, nbits=pq_nbits)
            return PQSentenceTransformer(path, truncate_dim=truncate_dim, M=pq_M, nbits=pq_nbits, model_kwargs=kwargs)

        # 4. TurboQuant Models
        elif tq_bits is not None:
            print(f"Loading {model_name} as TurboQuantSentenceTransformer ({tq_bits} bits)")
            if modules:
                return TurboQuantSentenceTransformer(truncate_dim=truncate_dim, modules=modules, tq_bits=tq_bits)
            return TurboQuantSentenceTransformer(path, truncate_dim=truncate_dim, tq_bits=tq_bits, model_kwargs=kwargs)

        # 5. Standard Models
        else:
            print(f"Using SentenceTransformer for {model_name}")
            if modules:
                return SentenceTransformer(modules=modules, truncate_dim=truncate_dim)
            return SentenceTransformer(path, truncate_dim=truncate_dim, model_kwargs=kwargs)

    # --- Logic A: Custom Pooling Mode Requested ---
    if "pooling_mode" in model_info:
        print(f"Loading {model_name} with custom '{model_info['pooling_mode']}' pooling...")
        
        # 1. Load the base transformer
        transformer_layer = s_models.Transformer(model_path)
        
        # 2. Create the pooling layer manually
        pooling_layer = s_models.Pooling(
            word_embedding_dimension=transformer_layer.get_word_embedding_dimension(),
            pooling_mode=model_info["pooling_mode"],
        )
        
        # 3. Instantiate using the list of modules
        return _instantiate_model(modules=[transformer_layer, pooling_layer])

    # --- Logic B: Default Pooling / Loading ---
    else:
        print(f"Loading {model_name} with default pooling...")
        return _instantiate_model(path=model_path, kwargs=model_kwargs)


def add_aggregrated_emb_info(emb_info, targets=["mean", "weightedmean"]):
    global dataset_name
    for model_name in emb_info:
        aggregated_info = {
            "queries": {
                "n": 0,
                "dimension": 0,
                "element_size_bit": 0,
            },
            "corpus": {
                "n": 0,
                "dimension": 0,
                "element_size_bit": 0,
            },
        }

        for evaluator_name, evaluator_info in emb_info[model_name].items():

            # parse evaluator_name for subset
            subset_name = evaluator_name.split("__")[1]
            if "mean" in subset_name:
                continue  # skip mean evaluators
        
            query_info = evaluator_info.get("queries", {})
            corpus_info = evaluator_info.get("corpus", {})

            aggregated_info["queries"]["n"] += query_info.get("n", 0)
            aggregated_info["queries"]["dimension"] = query_info.get("dimension", 0)
            aggregated_info["queries"]["element_size_bit"] = query_info.get("element_size_bit", 0)

            aggregated_info["corpus"]["n"] += corpus_info.get("n", 0)
            aggregated_info["corpus"]["dimension"] = corpus_info.get("dimension", 0)
            aggregated_info["corpus"]["element_size_bit"] = corpus_info.get("element_size_bit", 0)

        for target in targets:
            emb_info[model_name][f"{dataset_name}__{target}____evaluator"] = aggregated_info


def evaluate():
    time_taken = load_json_if_exists(json_time_path)
    all_results = load_json_if_exists(json_output_path)
    emb_info = load_json_if_exists(emb_info_path)
    query_counts = {}

    calibration_texts = get_calibration_corpus_texts(n_samples=1000)
    # calibration_texts = []

    subsets = dataset_config[dataset_name].get("subsets", [None])
    for subset in subsets:
        # this will loop multiple times if subsets are provided else it will loop once
        # check if there is multiple data_files for the dataset_name

        # precompute corpus embeddings for different dataset_name-subset-model_name
        # for different data_files, only relevant_docs / qrels are different
        corpus_pre_computed_embeddings, time_taken = pre_compute_corpus_embedding(
            models,
            dataset_name,
            subset,
            all_results,
            dataset_config,
            time_taken,
            calibration_texts,
            embedding_cache_dir=embedding_cache_dir,
        )
        for data_file in dataset_config[dataset_name].get("data_files", [None]):
            corpus, queries, relevant_docs = get_dataset(
                dataset_name,
                subset,
                relevant_docs_split="test" if data_file is None else "train",
                data_file=data_file,
            )
            if relevant_docs is not None:
                query_counts[
                    f"{dataset_name}__{subset if subset is not None else ''}__"
                    f"{data_file if data_file is not None else ''}"
                ] = len(relevant_docs)
            evaluators = get_evaluator(
                dataset_name,
                queries,
                corpus,
                relevant_docs,
                subset,
                data_file,
                dataset_config=dataset_config[dataset_name]
            )
            # Looping models
            for model_name, model_info in models.items():
                print(
                    f"Evaluating model: {model_name}, subset {subset} and data_file: {data_file}",
                )
                if check_if_eval_already_exists(
                    all_results,
                    model_name,
                    subset,
                    data_file,
                ):
                    print(
                        f"Model {model_name} with the subset {subset} and data_file "
                        f"{data_file} already evaluated. Skipping...",
                    )
                    continue
                model = load_model_with_proper_pooling(model_name, model_info, calibration_texts=calibration_texts)
                evaluator_obj = evaluators.get(model_info.get("similarity_fn_name", "cosine"))
                results = evaluator_obj(
                    model,
                    query_prompt_str=model_info.get("query_prompt", None),
                    corpus_embeddings=corpus_pre_computed_embeddings.get(
                        f"{dataset_name}__{subset}__{model_name}"
                    ),
                )
                results = {
                    k: v for k, v in results.items() if k.startswith(dataset_name)
                }  # filtering out non compatible keys
                if model_name not in all_results:
                    all_results[model_name] = {}
                all_results[model_name] = {**all_results[model_name], **results}

                import copy
                # for emb_info
                if type(evaluator_obj) is MultiGPUNanoBEIREvaluator:
                    # in this case the return can be appended to emb_info directly
                    emb_info[model_name] = copy.deepcopy(getattr(evaluator_obj, "last_embedding_info", {}))
                else:
                    # Pick up emb info produced by the evaluator
                    if hasattr(evaluator_obj, "last_embedding_info"):
                        if model_name not in emb_info:
                            emb_info[model_name] = {}
                        emb_info[model_name][evaluator_obj.name] = copy.deepcopy(getattr(evaluator_obj, "last_embedding_info", {}))
    

    if len(dataset_config[dataset_name].get("paths", [])) > 1:
        # computing query counts for different paths
        query_counts = generate_query_counts_for_nanobeir(dataset_config, dataset_name)

    if len(subsets) > 1 or len(dataset_config[dataset_name].get("paths", [])) > 1:
        # need to add a mean of metrics from different subsets of different models
        add_mean_metrics(all_results, query_counts, mean_basis="subset")
    elif len(dataset_config[dataset_name].get("data_files", [])) > 1:
        # need to add a mean of metrics from different data_files of different models
        add_mean_metrics(all_results, query_counts, mean_basis="data_file")

    # also add aggregrated emb_info if multiple subsets or data_files as mean and weighted mean
    if (len(subsets) > 1 or len(dataset_config[dataset_name].get("paths", [])) > 1) or len(dataset_config[dataset_name].get("data_files", [])) > 1:
        add_aggregrated_emb_info(emb_info)

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    with open(json_time_path, "w", encoding="utf-8") as f:
        json.dump(time_taken, f, ensure_ascii=False, indent=4)

    with open(emb_info_path, "w", encoding="utf-8") as f:
        json.dump(emb_info, f, ensure_ascii=False, indent=4)
    print(f"Saved embedding info to {emb_info_path}")


def convert_json_output_to_df(json_output_path):
    if not os.path.exists(json_output_path):
        print(
            f"JSON output path {json_output_path} does not exist."
            "Please run the evaluation first.",
        )
        return
    with open(json_output_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    # lets make a df first of the json
    records = []
    for model, metrics in all_results.items():
        for metric_name_full, value in metrics.items():
            parts = metric_name_full.split("__")
            dataset_name_str = parts[0]
            subset_str = parts[1]
            data_file_str = parts[2]
            remaining_str = parts[-1]

            parts = remaining_str.split("@")
            k = int(parts[1])
            metric_name = parts[0].split("_")[-1]
            records.append(
                {
                    "dataset_name": dataset_name_str,
                    "subset": subset_str,
                    "data_file": data_file_str,
                    "model": model,
                    "metric": metric_name,
                    "k": k,
                    "value": value,
                },
            )

    # Create a pandas DataFrame
    df = pd.DataFrame(records)

    return df


def load_and_simplify_model_size_info(emb_info_path):
    if not os.path.exists(emb_info_path):
        print(
            f"Embedding info path {emb_info_path} does not exist."
            "Please run the evaluation first.",
        )
        return {}
    with open(emb_info_path, "r", encoding="utf-8") as f:
        emb_info = json.load(f)

    simplified_info = {}
    df_data = []

    for model_name, model_info in emb_info.items():
        simplified_info[model_name] = {}
        
        # loop through evaluator names
        for evaluator_name, evaluator_info in model_info.items():
            # compute total index bit for queries
            query_info = evaluator_info.get("queries", {})
            total_query_bits = query_info.get("n", 0) * query_info.get("dimension", 0) * query_info.get("element_size_bit", 0)
            # compute total index bit for corpus
            corpus_info = evaluator_info.get("corpus", {})
            total_corpus_bits = corpus_info.get("n", 0) * corpus_info.get("dimension", 0) * corpus_info.get("element_size_bit", 0)

            # parsing evaluator name to find dataset_name, subset, data_file
            parts = evaluator_name.split("__")
            dataset_name_str = parts[0]
            subset_str = parts[1]
            data_file_str = parts[2]

            df_data.append({
                "dataset_name": dataset_name_str,
                "subset": subset_str,
                "data_file": data_file_str,
                "model": model_name,
                "total_embedding_bits": total_query_bits + total_corpus_bits,
                "query_bits": total_query_bits,
                "corpus_bits": total_corpus_bits,
                "n_query": query_info.get("n", 0),
                "n_corpus": corpus_info.get("n", 0),
                "dim_query_emb": query_info.get("dimension", 0),
                "dim_corpus_emb": corpus_info.get("dimension", 0),
                "element_size_bit_query": query_info.get("element_size_bit", 0),
                "element_size_bit_corpus": corpus_info.get("element_size_bit", 0),
            })

    return pd.DataFrame(df_data)


_METRIC_LABELS = {
    "mrr": "MRR", "ndcg": "nDCG", "accuracy": "Accuracy",
    "precision": "Precision", "recall": "Recall", "map": "MAP",
}


def _setup_plot_style():
    """Apply a clean, publication-ready style shared across all plot functions."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.title_fontsize": 10,
        "figure.titlesize": 13,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _get_marker_for_model(model_key):
    """Return a distinct marker shape based on model group for scatter plots.

    Marker legend:
      o  — baseline / finetuned (no compression)
      s  — post-quantization (int8 / binary applied after training)
      D  — BAT (binary-aware training)
      ^  — MRL truncation only
      v  — MRL + post-binary
      P  — PQ only (full dim, no MRL)
      *  — MRL + PQ
    """
    if "bat" in model_key:
        return "D"
    if "mrl" in model_key and "post-binary" in model_key:
        return "v"
    if "post-int8" in model_key or ("post-binary" in model_key and "mrl" not in model_key):
        return "s"
    if "mrl" in model_key and "pq" in model_key:
        return "*"
    if "pq" in model_key:
        return "P"
    if "mrl" in model_key:
        return "^"
    if "tq" in model_key:
        return "h"   # hexagon — TurboQuant
    return "o"


def plot_performance_to_size_eval(
    json_output_path,
    emb_info_path,
    top_ks=[1, 3, 5],
    model_name_mapper=None,
    dpi=300
):
    _setup_plot_style()

    # 1. Load & filter data
    df = convert_json_output_to_df(json_output_path)
    df = df[df["metric"].isin(desired_metric_types)]
    df = df[df["k"].isin(top_ks)]
    if df.empty:
        print(f"No data found for K values: {top_ks}.")
        return

    # 2. Load embedding size info and merge
    emb_size_df = load_and_simplify_model_size_info(emb_info_path)
    merged_df = pd.merge(
        df, emb_size_df,
        on=["dataset_name", "subset", "data_file", "model"],
        how="inner",
    )
    if model_name_mapper:
        merged_df["model_display_name"] = merged_df["model"].map(
            lambda x: model_name_mapper.get(x, x)
        )
    else:
        merged_df["model_display_name"] = merged_df["model"]
    merged_df["total_embedding_mb"] = merged_df["total_embedding_bits"] / 8.0 / (1024 ** 2)

    # 3. Output dir
    _dir_to_save = os.path.join(output_dir_plots, dataset_name, "performance_to_size_scatter_plots")
    os.makedirs(_dir_to_save, exist_ok=True)
    merged_df.to_csv(os.path.join(_dir_to_save, "performance_to_size_data.csv"), index=False)

    # 4. One plot per (subset, data_file, metric, k)
    for subset in merged_df["subset"].unique():
        subset_df = merged_df[merged_df["subset"] == subset]
        for data_file in subset_df["data_file"].unique():
            df_filt = subset_df[subset_df["data_file"] == data_file]
            for metric in df_filt["metric"].unique():
                metric_df = df_filt[df_filt["metric"] == metric]
                for k in metric_df["k"].unique():
                    plot_df = metric_df[metric_df["k"] == k].copy()

                    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

                    # Plot each model with its own color + marker shape
                    legend_handles = []
                    for model_key in plot_df["model"].unique():
                        row = plot_df[plot_df["model"] == model_key].iloc[0]
                        color = models.get(model_key, {}).get("color", "#555555")
                        marker = _get_marker_for_model(model_key)
                        ax.scatter(
                            row["total_embedding_mb"], row["value"],
                            color=color, marker=marker,
                            s=90, edgecolors="white", linewidths=0.8, zorder=3,
                        )
                        legend_handles.append(
                            plt.Line2D(
                                [0], [0], marker=marker, color="w",
                                markerfacecolor=color, markersize=8,
                                markeredgecolor="white",
                                label=row["model_display_name"],
                            )
                        )

                    ax.set_xlabel("Total Embedding Size (MB)", labelpad=8)
                    ax.set_ylabel(
                        f"{_METRIC_LABELS.get(metric.lower(), metric.upper())} @ {k}",
                        labelpad=8,
                    )
                    title_subset = subset if subset else "all"
                    ax.set_title(
                        f"Performance vs. Embedding Size\n"
                        f"{dataset_name}  |  subset: {title_subset}",
                        pad=12,
                    )
                    sns.despine(ax=ax, top=True, right=True)
                    ax.yaxis.set_major_formatter(
                        plt.FuncFormatter(lambda v, _: f"{v:.3f}")
                    )

                    ax.legend(
                        handles=legend_handles,
                        loc="upper left",
                        bbox_to_anchor=(1.01, 1.0),
                        borderaxespad=0,
                        frameon=True,
                        framealpha=0.95,
                        edgecolor="#dddddd",
                        fontsize=8,
                    )

                    plot_filename = (
                        f"performance_vs_size__subset_{subset}__datafile_{data_file}__"
                        f"metric_{metric}__k_{k}.png"
                    ).replace("/", "_").replace(" ", "_")
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(_dir_to_save, plot_filename),
                        dpi=dpi, bbox_inches="tight",
                    )
                    plt.close()
                    print(f"Saved plot to {os.path.join(_dir_to_save, plot_filename)}")


def plot_results(
    json_output_path,
    top_ks=[1, 3, 5, 10],
    dpi=300,
    legend_cols=2,
    model_name_mapper=None,
):
    """Plots grouped bar charts of model performance metrics across K values."""
    _setup_plot_style()
    print(f"Plotting results for K values: {top_ks}...")

    # 1. Load & filter data
    df = convert_json_output_to_df(json_output_path)
    df = df[df["metric"].isin(desired_metric_types)]
    df = df[df["k"].isin(top_ks)]
    if df.empty:
        print(f"No data found for K values: {top_ks}.")
        return

    all_model_configs = {**models}
    model_color_palette = {name: cfg["color"] for name, cfg in all_model_configs.items()}
    model_hatch_palette = {name: cfg.get("hatch", "") for name, cfg in all_model_configs.items()}

    present_models = set(df["model"].unique())
    sorted_model_names = [m for m in all_model_configs.keys() if m in present_models]

    for subset in df["subset"].unique():
        subset_df = df[df["subset"] == subset]
        for data_file in subset_df["data_file"].unique():
            plot_df = subset_df[subset_df["data_file"] == data_file]

            n_metrics = plot_df["metric"].nunique()
            g = sns.catplot(
                data=plot_df,
                x="k", y="value", hue="model",
                hue_order=sorted_model_names,
                col="metric", kind="bar",
                col_wrap=min(2, n_metrics), sharey=False,
                legend=False,
                palette=model_color_palette,
                height=5, aspect=1.4,
                edgecolor="#444444",
                linewidth=0.5,
            )

            # Apply hatches
            for ax in g.axes.flat:
                for i, container in enumerate(ax.containers):
                    if i < len(sorted_model_names):
                        hatch = model_hatch_palette.get(sorted_model_names[i], "")
                        for bar in container:
                            bar.set_hatch(hatch)

            # Nicely formatted subplot titles
            for ax in g.axes.flat:
                col_val = ax.get_title()
                ax.set_title(
                    _METRIC_LABELS.get(col_val.lower(), col_val.upper()),
                    fontsize=12, pad=8,
                )
            g.set_axis_labels("K", "Score")
            g.set(ylim=(0, None))
            sns.despine(fig=g.fig, top=True, right=True)

            # Legend with display names
            legend_handles = [
                Patch(
                    facecolor=model_color_palette[m],
                    hatch=model_hatch_palette.get(m, ""),
                    label=model_name_mapper.get(m, m) if model_name_mapper else m,
                    edgecolor="#444444",
                    linewidth=0.5,
                )
                for m in sorted_model_names
            ]

            title = f"Model Performance — {dataset_name}"
            if subset and str(subset) not in ("None", ""):
                title += f" / {subset}"
            if data_file and str(data_file) not in ("None", ""):
                title += f"\n{data_file.split('/')[-1]}"

            g.fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
            g.fig.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.01),
                ncol=legend_cols,
                frameon=False,
                fontsize=9,
                columnspacing=1.2,
                handlelength=2.0,
            )

            os.makedirs(os.path.join(output_dir_plots, dataset_name), exist_ok=True)
            out_file = os.path.join(
                output_dir_plots, dataset_name,
                f"{dataset_name}_{subset}_{data_file.split('/')[-1] if data_file else 'all'}.png",
            )
            plt.savefig(out_file, dpi=dpi, bbox_inches="tight")
            plt.close(g.fig)
            print(f"Saved plot to {out_file}")

def plot_data_files_based_eval(json_output_path, top_ks=[1, 3, 5, 10]):
    """Bar charts per model, comparing performance across data files."""
    _setup_plot_style()
    print("Plotting data files based evaluation...")

    df = convert_json_output_to_df(json_output_path)
    df = df[df["metric"].isin(desired_metric_types)]
    df = df[~df["data_file"].str.contains("mean")]

    if df["data_file"].nunique() <= 1:
        return

    output_dir_plots_ = os.path.join(output_dir_plots, dataset_name, "data_files_based_eval")
    os.makedirs(output_dir_plots_, exist_ok=True)

    data_files = dataset_config[dataset_name].get("data_files", [])
    data_files_colors = dataset_config[dataset_name].get("data_files_colors", [])
    # Fall back to a default hatch cycle if none configured
    _default_hatches = ["//", "\\\\", "xx", "..", "--", "||", "++", "oo"] * 4
    data_files_hatches = dataset_config[dataset_name].get("data_files_hatches", _default_hatches)

    dataset_color_palette = dict(zip(data_files, data_files_colors))
    dataset_hatch_palette = dict(zip(data_files, data_files_hatches))

    for model_name in df["model"].unique():
        df_model = df[df["model"] == model_name]
        sorted_data_file_names = sorted(df_model["data_file"].unique())

        g = sns.catplot(
            data=df_model,
            x="k", y="value", hue="data_file",
            hue_order=sorted_data_file_names,
            col="metric", kind="bar",
            col_wrap=2, sharey=False,
            legend=False,
            palette=dataset_color_palette,
            height=5, aspect=1.8,
            edgecolor="#444444",
            linewidth=0.5,
        )

        # Apply hatches
        for ax in g.axes.flat:
            for i, container in enumerate(ax.containers):
                if i < len(sorted_data_file_names):
                    hatch = dataset_hatch_palette.get(sorted_data_file_names[i], "")
                    for bar in container:
                        bar.set_hatch(hatch)

        # Nicely formatted subplot titles
        for ax in g.axes.flat:
            col_val = ax.get_title()
            ax.set_title(
                _METRIC_LABELS.get(col_val.lower(), col_val.upper()),
                fontsize=12, pad=8,
            )
        g.set_axis_labels("K", "Score")
        g.set(ylim=(0, None))
        sns.despine(fig=g.fig, top=True, right=True)

        legend_handles = [
            Patch(
                facecolor=dataset_color_palette.get(df_name, "#888888"),
                hatch=dataset_hatch_palette.get(df_name, ""),
                label=df_name.split("/")[-1].replace(".tsv", ""),
                edgecolor="#444444",
                linewidth=0.5,
            )
            for df_name in sorted_data_file_names
        ]

        display_name = models.get(model_name, {}).get("display_name", model_name)
        g.fig.suptitle(
            f"{display_name}\n{dataset_name}",
            fontsize=13, fontweight="bold", y=1.02,
        )
        g.fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=min(4, len(sorted_data_file_names)),
            frameon=False,
            fontsize=9,
            title="Data File",
            title_fontsize=9,
            columnspacing=1.2,
        )

        out_file = os.path.join(output_dir_plots_, f"{model_name}_performance_plots.png")
        plt.savefig(out_file, bbox_inches="tight", dpi=300)
        plt.close(g.fig)
        print(f"Saved plot to {out_file}")


if __name__ == "__main__":
    if not just_plot:
        evaluate()

    model_name_mapper = {
        model_name: model_info["display_name"]
        for model_name, model_info in models.items()
        if "display_name" in model_info
    }

    plot_results(json_output_path, plotks, model_name_mapper=model_name_mapper)
    plot_data_files_based_eval(json_output_path, plotks)

    plot_performance_to_size_eval(json_output_path,
                                  emb_info_path,
                                  model_name_mapper=model_name_mapper)

    # --- Interactive HTML dashboard ---
    from interactive_plot import generate_interactive_html
    generate_interactive_html(
        json_output_path=json_output_path,
        emb_info_path=emb_info_path,
        dataset_name=dataset_name,
        output_dir=interactive_plot_path,
        models_meta={k: v for k, v in models.items()},
    )