from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetSpec:
    loader: str                        # "nanobeir" | "beir" | "hf_ir"
    # NanoBEIR / multi-subset HF datasets
    subsets: Optional[list[str]] = None          # None = use all
    hf_prefix: Optional[str] = None             # e.g. "zeta-alpha-ai/Nano"
    # BEIR-style local datasets
    dataset_cache_path: Optional[str] = None
    # Single HF dataset with splits (NASA-style)
    hf_path: Optional[str] = None
    data_files: Optional[list[str]] = None       # qrel file names inside HF dataset
    data_file_colors: Optional[list[str]] = None # per-data-file plot color


DATASETS: dict[str, DatasetSpec] = {
    "nanobeir": DatasetSpec(
        loader="nanobeir",
        subsets=[
            "NanoClimateFEVER",
            "NanoDBPedia",
            "NanoFEVER",
            "NanoFiQA2018",
            "NanoHotpotQA",
            "NanoMSMARCO",
            "NanoNFCorpus",
            "NanoNQ",
            "NanoQuoraRetrieval",
            "NanoSCIDOCS",
            "NanoArguAna",
            "NanoSciFact",
            "NanoTouche2020",
        ],
        hf_prefix="zeta-alpha-ai/Nano",
    ),
    "beir": DatasetSpec(
        loader="beir",
        subsets=[
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
        dataset_cache_path="./beir_datasets",
    ),
    "nasa_smd_ir": DatasetSpec(
        loader="hf_ir",
        hf_path="nasa-impact/nasa-smd-IR-benchmark",
    ),
    "nasa_sde_ir_v3": DatasetSpec(
        loader="hf_ir",
        hf_path="nasa-impact/nasa-sde-IR-benchmark-sample-v3",
        data_files=[
            "qrels/question-answer~SDE_general_v2.tsv",
            "qrels/question-answer~SDE_general_v3.tsv",
            "qrels/search_term-document~CMR.tsv",
            "qrels/search_term-document~PDS.tsv",
            "qrels/search_term-document~SDE_general_v2.tsv",
            "qrels/search_term-document~SDE_general_v3.tsv",
            "qrels/title-description~CMR.tsv",
            "qrels/title-description~PDS.tsv",
        ],
        data_file_colors=[
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        ],
    ),
    "nasa_sde_ir_20251024_v5": DatasetSpec(
        loader="hf_ir",
        hf_path="nasa-impact/nasa-sde-IR-benchmark-20251024-v5",
        data_files=[
            "qrels/qa_pairs.tsv",
            "qrels/search_pairs.tsv",
        ],
        data_file_colors=["#1f77b4", "#ff7f0e"],
    ),
}


# NanoBEIR subset → HuggingFace dataset ID
NANOBEIR_SUBSET_TO_HF: dict[str, str] = {
    subset: f"zeta-alpha-ai/{subset}"
    for subset in DATASETS["nanobeir"].subsets
}
