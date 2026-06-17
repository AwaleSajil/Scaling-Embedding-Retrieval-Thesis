#!/usr/bin/env python3
"""
eval_v2 entry point.

Usage:
    cd /nas/rhome/sawale/thesis
    python -m eval_v2.run --dataset nanobeir --output_dir eval_v2/outputs/results/

The script will:
  1. For each NanoBEIR subset, load corpus / queries / qrels from HuggingFace.
  2. For each unique model checkpoint, encode corpus and query embeddings once
     (with multi-GPU pool) and cache them on disk.
  3. For each model, apply transforms (truncate / binarize / quantize / PQ / TQ)
     and run per-query IR evaluation.
  4. Compute significance tests across model pairs.
  5. Build a self-contained interactive HTML dashboard.
"""
import argparse
import os
import sys
from pathlib import Path

# Allow running as `python eval_v2/run.py` from the thesis root
_THESIS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_THESIS_ROOT))

# Load .env from the thesis root so HUGGINGFACE_TOKEN and other secrets are available
from dotenv import load_dotenv
load_dotenv(_THESIS_ROOT / ".env")

from eval_v2.config.models import MODELS, ModelSpec
from eval_v2.config.datasets import DATASETS, NANOBEIR_SUBSET_TO_HF
from eval_v2.core.cache import EmbeddingCache
from eval_v2.core.evaluator import run_ir_eval
from eval_v2.core.significance import compute_significance
from eval_v2.results.store import ResultsStore
from eval_v2.html.builder import build_html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="eval_v2 — modular retrieval evaluation")
    p.add_argument("--dataset", default="nanobeir",
                   choices=list(DATASETS.keys()),
                   help="Dataset to evaluate on.")
    p.add_argument("--output_dir", default="eval_v2/outputs/results/",
                   help="Directory for aggregate/per-query/emb-info/significance JSONs.")
    p.add_argument("--cache_dir", default="eval_v2/outputs/cache/",
                   help="Directory for cached raw float32 embeddings.")
    p.add_argument("--html_path", default=None,
                   help="Path for the HTML dashboard. Defaults to <output_dir>/<dataset>/index.html "
                        "(named index.html so it can be served directly as a static site, e.g. a HF Space)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--chunk_size", type=int, default=1000,
                   help="Encoding chunk size passed to SentenceTransformer.encode().")
    p.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10],
                   help="K values for IR metrics.")
    p.add_argument("--models", nargs="*", default=None,
                   help="Subset of model keys to evaluate. Defaults to all models.")
    p.add_argument("--subsets", nargs="*", default=None,
                   help="Subset of dataset subsets. Defaults to all.")
    p.add_argument("--just_html", action="store_true",
                   help="Skip evaluation; only rebuild the HTML dashboard from existing results.")
    p.add_argument("--run_significance", action="store_true",
                   help="Force recomputation of all significance tests, ignoring cached results.")
    p.add_argument("--hf_token", default=None,
                   help="HuggingFace access token (or set HUGGINGFACE_TOKEN env var).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_nanobeir_subset(subset_name: str, hf_token: str | None = None):
    """
    Load corpus, queries, qrels for one NanoBEIR subset from HuggingFace.
    Returns:
        corpus_ids   : list[str]
        corpus_texts : list[str]
        query_ids    : list[str]
        query_texts  : list[str]
        qrels        : dict[str, set[str]]   (query_id → set of relevant corpus_ids)
    """
    from datasets import load_dataset

    hf_path = NANOBEIR_SUBSET_TO_HF[subset_name]
    kw = dict(token=hf_token) if hf_token else {}

    corpus_ds = load_dataset(hf_path, "corpus", split="train", **kw)
    queries_ds = load_dataset(hf_path, "queries", split="train", **kw)
    qrels_ds = load_dataset(hf_path, "qrels", split="train", **kw)

    corpus_ids, corpus_texts = [], []
    for row in corpus_ds:
        if row["text"].strip():
            corpus_ids.append(row["_id"])
            corpus_texts.append(row["text"])

    query_ids, query_texts = [], []
    for row in queries_ds:
        if row["text"].strip():
            query_ids.append(row["_id"])
            query_texts.append(row["text"])

    qrels: dict[str, set[str]] = {}
    for row in qrels_ds:
        qid = row["query-id"]
        qrels.setdefault(qid, set()).add(row["corpus-id"])

    return corpus_ids, corpus_texts, query_ids, query_texts, qrels


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_nanobeir(args, store: ResultsStore, cache: EmbeddingCache):
    spec = DATASETS[args.dataset]
    subsets = args.subsets or spec.subsets

    # Filter models
    model_keys = args.models or list(MODELS.keys())
    models_to_run = {k: MODELS[k] for k in model_keys if k in MODELS}

    hf_token = args.hf_token or os.environ.get("HUGGINGFACE_TOKEN")

    for subset in subsets:
        print(f"\n{'='*60}")
        print(f"  Subset: {subset}")
        print(f"{'='*60}")

        corpus_ids, corpus_texts, query_ids, query_texts, qrels = \
            load_nanobeir_subset(subset, hf_token)

        print(f"  Corpus: {len(corpus_ids)}  Queries: {len(query_ids)}")

        # --- Cache embeddings per unique model path ---
        # Group models by base checkpoint so we encode once per path.
        path_to_keys: dict[str, list[str]] = {}
        for mkey, mspec in models_to_run.items():
            path_to_keys.setdefault(mspec.path, []).append(mkey)

        for model_path, mkeys in path_to_keys.items():
            print(f"\n  [Encode] {model_path}")
            # Use any spec from this group for pooling_mode etc.
            any_spec = models_to_run[mkeys[0]]

            cache.get_or_compute(
                model_path=model_path,
                texts=corpus_texts,
                dataset=args.dataset,
                subset=subset,
                split="corpus",
                spec=any_spec,
            )
            cache.get_or_compute(
                model_path=model_path,
                texts=query_texts,
                dataset=args.dataset,
                subset=subset,
                split="queries",
                spec=any_spec,
            )

        # --- Evaluate each model ---
        for mkey, mspec in models_to_run.items():
            if store.already_evaluated(mkey, subset):
                print(f"  [Skip] {mkey} / {subset} already done")
                continue

            print(f"\n  [Eval] {mkey}")

            corpus_embs_raw = cache.get_or_compute(
                model_path=mspec.path,
                texts=corpus_texts,
                dataset=args.dataset,
                subset=subset,
                split="corpus",
                spec=mspec,
            )
            query_embs_raw = cache.get_or_compute(
                model_path=mspec.path,
                texts=query_texts,
                dataset=args.dataset,
                subset=subset,
                split="queries",
                spec=mspec,
            )

            result = run_ir_eval(
                dataset=args.dataset,
                subset=subset,
                corpus_ids=corpus_ids,
                corpus_embs_raw=corpus_embs_raw,
                query_ids=query_ids,
                query_texts=query_texts,
                query_embs_raw=query_embs_raw,
                qrels=qrels,
                spec=mspec,
                ks=args.ks,
            )

            store.save_subset_result(mkey, result)
            agg_str = "  ".join(
                f"{m}={v:.4f}"
                for m, v in sorted(result.aggregate.items())
                if "@10" in m
            )
            print(f"    {agg_str}")

    # --- Mean across subsets ---
    print("\n[Mean] Computing mean metrics across subsets ...")
    for mkey in models_to_run:
        store.save_mean_metrics(mkey, subsets)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Each dataset gets its own subdirectory so files never collide
    dataset_output_dir = os.path.join(args.output_dir, args.dataset)

    if args.html_path is None:
        args.html_path = os.path.join(dataset_output_dir, "index.html")

    store = ResultsStore(dataset_output_dir)
    cache = EmbeddingCache(
        root=args.cache_dir,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )

    if not args.just_html:
        if args.dataset == "nanobeir":
            evaluate_nanobeir(args, store, cache)
        else:
            raise NotImplementedError(
                f"Dataset '{args.dataset}' loader not yet implemented in eval_v2. "
                "Add a loader in run.py following the nanobeir pattern."
            )

        # Significance testing
        existing_sig = {} if args.run_significance else store.load_significance()
        if existing_sig:
            print(f"\n[Significance] {len(existing_sig)} existing entries found; computing only new pairs ...")
        else:
            print("\n[Significance] Computing pairwise significance tests ...")
        new_sig = compute_significance(store, existing=existing_sig)
        merged_sig = {**existing_sig, **new_sig}
        store.save_significance(merged_sig)
        print(f"  {len(new_sig)} new pairs computed, {len(merged_sig)} total.")

    # Build HTML
    print(f"\n[HTML] Building dashboard → {args.html_path}")
    build_html(store, args.html_path)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
