"""
IR evaluation engine for eval_v2.

Key difference from eval/custum_evals.py:
  - Returns per-query results (needed for significance testing).
  - Takes pre-computed (and already-transformed) corpus + query embeddings.
  - No inheritance from ST evaluator classes.
  - Multi-GPU pool is owned by the cache layer, not here.
  - PQ and TurboQuant fitting happens here (after transforms), since they
    need corpus-level data to fit the codebook / quantizer.
"""
import math
import heapq
from dataclasses import dataclass, field

import numpy as np
import torch

from eval_v2.config.models import ModelSpec
from eval_v2.wrappers.base import hamming_similarity, embedding_info


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query_id: str
    query_text: str
    relevant_ids: set
    ranked_ids: list[str]      # length = max_k, in descending score order
    ranked_scores: list[float]
    metrics: dict[str, float]  # "ndcg@10", "mrr@10", "accuracy@1", etc.


@dataclass
class SubsetResult:
    dataset: str
    subset: str
    per_query: list[QueryResult]
    aggregate: dict[str, float]   # mean over queries
    emb_info: dict                # {"queries": {...}, "corpus": {...}}


# ---------------------------------------------------------------------------
# Per-query metric helpers
# ---------------------------------------------------------------------------

def _ndcg_at_k(ranked: list[str], relevant: set, k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, doc in enumerate(ranked[:k])
        if doc in relevant
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def _mrr_at_k(ranked: list[str], relevant: set, k: int) -> float:
    for i, doc in enumerate(ranked[:k]):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _accuracy_at_k(ranked: list[str], relevant: set, k: int) -> float:
    return float(any(doc in relevant for doc in ranked[:k]))


def _recall_at_k(ranked: list[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for doc in ranked[:k] if doc in relevant) / len(relevant)


def _compute_query_metrics(ranked: list[str], relevant: set, ks: list[int]) -> dict[str, float]:
    metrics = {}
    for k in ks:
        metrics[f"ndcg@{k}"] = _ndcg_at_k(ranked, relevant, k)
        metrics[f"mrr@{k}"] = _mrr_at_k(ranked, relevant, k)
        metrics[f"accuracy@{k}"] = _accuracy_at_k(ranked, relevant, k)
        metrics[f"recall@{k}"] = _recall_at_k(ranked, relevant, k)
    return metrics


# ---------------------------------------------------------------------------
# PQ / TurboQuant post-processing (applied to already-transformed tensors)
# ---------------------------------------------------------------------------

def _apply_pq(corpus_embs: torch.Tensor, query_embs: torch.Tensor, spec: ModelSpec):
    """Fit PQ index on corpus, reconstruct both corpus and queries (asymmetric PQ)."""
    import faiss
    corpus_np = corpus_embs.detach().cpu().numpy().astype(np.float32)
    d = corpus_np.shape[1]
    assert d % spec.pq_M == 0, f"Dim {d} not divisible by pq_M={spec.pq_M}"
    index = faiss.IndexPQ(d, spec.pq_M, spec.pq_nbits)
    index.train(corpus_np)
    # No index.add(): we only need sa_encode/sa_decode for the reconstruction, and
    # adding would keep a second full copy of the codes. On the 5.4M-doc BEIR
    # subsets (fever, climate-fever, hotpotqa) the live arrays here are already
    # ~17 GB each, so the redundant copy matters.
    codes = index.sa_encode(corpus_np)
    del corpus_np
    corpus_recon = torch.from_numpy(index.sa_decode(codes))
    del codes
    # Queries stay float32 (asymmetric PQ)
    query_float = query_embs.float() if not query_embs.is_floating_point() else query_embs
    return corpus_recon, query_float, index


def _apply_tq(corpus_embs: torch.Tensor, query_embs: torch.Tensor, spec: ModelSpec):
    """Quantize + dequantize with TurboQuantMSE (lazy-init per dim)."""
    from turboquant import TurboQuantMSE
    dim = corpus_embs.shape[1]
    tq = TurboQuantMSE(dim=dim, bits=spec.tq_bits, device="cpu")
    def _quant(t: torch.Tensor) -> torch.Tensor:
        t = t.cpu().float()
        idx, norms = tq.quantize(t)
        return tq.dequantize(idx, norms)
    return _quant(corpus_embs), _quant(query_embs)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _to_float_tensor(x) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not x.is_floating_point():
        x = x.float()
    return x


def _prepare_queries(q_embs: torch.Tensor, similarity: str) -> torch.Tensor:
    """Query-side work that does not depend on the corpus chunk.

    Hoisted out of the chunk loop: this used to run inside _score_chunk, so on
    full BEIR the same (n_queries x 768) normalize was recomputed once per
    chunk -- 17,684 times per model on msmarco.
    """
    if similarity == "hamming":
        return q_embs
    return torch.nn.functional.normalize(q_embs.float(), dim=1)


def _score_chunk(q_prepared: torch.Tensor, c_chunk: torch.Tensor,
                 similarity: str) -> torch.Tensor:
    """Score one corpus chunk. *q_prepared* must come from _prepare_queries."""
    if similarity == "hamming":
        return hamming_similarity(q_prepared, c_chunk)
    # cosine — queries are already normalized
    c_norm = torch.nn.functional.normalize(c_chunk.float(), dim=1)
    return torch.mm(q_prepared, c_norm.t())


def _auto_chunk_size(n_queries: int, budget_bytes: int = 1_500_000_000) -> int:
    """Corpus rows to score at once, sized so the scores matrix fits a budget.

    The per-chunk Python cost (heap pushes, .tolist(), tensor setup) scales with
    the NUMBER of chunks, not corpus size -- taking top-k per query costs the
    same for a 500-row chunk as a 50,000-row one. So chunks should be as large
    as memory allows. The binding constraint is the scores matrix, which is
    n_queries x chunk float32.

    Measured on 16 threads: chunk=50,000 with the normalize hoisted is ~3.6x
    faster end-to-end than the previous fixed 500.
    """
    if n_queries <= 0:
        return 50_000
    rows = budget_bytes // (4 * n_queries)
    return int(max(2_048, min(100_000, rows)))


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_ir_eval(
    dataset: str,
    subset: str,
    corpus_ids: list[str],
    corpus_embs_raw: np.ndarray,          # float32 from cache, pre-transform
    query_ids: list[str],
    query_texts: list[str],
    query_embs_raw: np.ndarray,           # float32 from cache, pre-transform
    qrels: dict[str, set[str]],           # query_id → set of relevant corpus_ids
    spec: ModelSpec,
    ks: list[int] = [1, 3, 5, 10],
    corpus_chunk_size: int | None = None,
) -> SubsetResult:
    """
    Full IR evaluation pipeline for one (model, subset) pair.

    Steps:
      1. Apply corpus + query transforms (truncate/binarize/quantize).
      2. Apply PQ or TurboQuant if required (corpus-level fitting).
      3. Score all (query, corpus) pairs in chunks.
      4. Compute per-query metrics.
      5. Aggregate across queries.
      6. Return SubsetResult.
    """
    from eval_v2.wrappers.transforms import apply_transforms

    # 1. Apply transforms
    # For INT8/INT4: derive calibration from corpus so both sides use the same scale.
    calibration = None
    if spec.quant_bits in (8, 4):
        calib_raw = np.asarray(corpus_embs_raw, dtype=np.float32)
        if spec.truncate_dim is not None:
            calib_raw = calib_raw[:, : spec.truncate_dim]
        calibration = calib_raw[: min(1000, len(calib_raw))]

    corpus_embs = apply_transforms(corpus_embs_raw, spec, calibration_embeddings=calibration)
    query_embs = apply_transforms(query_embs_raw, spec, calibration_embeddings=calibration)

    # Record embedding info BEFORE PQ/TQ reshape corpus dims
    q_info = embedding_info(query_embs)
    c_info = embedding_info(corpus_embs)

    # 2a. PQ fitting
    pq_index = None
    if spec.pq_M is not None:
        corpus_embs_t = corpus_embs if isinstance(corpus_embs, torch.Tensor) else torch.from_numpy(corpus_embs)
        query_embs_t = query_embs if isinstance(query_embs, torch.Tensor) else torch.from_numpy(query_embs)
        corpus_embs, query_embs, pq_index = _apply_pq(corpus_embs_t, query_embs_t, spec)
        c_info["dimension"] = spec.pq_M
        c_info["element_size_bit"] = spec.pq_nbits

    # 2b. TurboQuant
    elif spec.tq_bits is not None:
        corpus_embs_t = corpus_embs if isinstance(corpus_embs, torch.Tensor) else torch.from_numpy(corpus_embs)
        query_embs_t = query_embs if isinstance(query_embs, torch.Tensor) else torch.from_numpy(query_embs)
        corpus_embs, query_embs = _apply_tq(corpus_embs_t, query_embs_t, spec)
        c_info["element_size_bit"] = spec.tq_bits
        q_info["element_size_bit"] = spec.tq_bits

    # 2c. Multi-bit ASigm — embeddings stay float32 at eval time but intended
    #     storage width is asigmoid_bits per dimension.
    elif spec.asigmoid_bits is not None:
        c_info["element_size_bit"] = spec.asigmoid_bits
        q_info["element_size_bit"] = spec.asigmoid_bits

    emb_info = {"queries": q_info, "corpus": c_info}

    # Ensure tensors
    if isinstance(corpus_embs, np.ndarray):
        corpus_embs = torch.from_numpy(corpus_embs)
    if isinstance(query_embs, np.ndarray):
        query_embs = torch.from_numpy(query_embs)

    n_queries = len(query_ids)
    max_k = max(ks)
    similarity = spec.similarity

    if corpus_chunk_size is None:
        corpus_chunk_size = _auto_chunk_size(n_queries)

    # Query-side prep is chunk-independent, so do it once rather than per chunk.
    q_prepared = _prepare_queries(query_embs, similarity)

    # 3. Score in chunks and track top-max_k per query
    # heap: list of heaps, one per query (min-heap of (score, corpus_id))
    heaps: list[list] = [[] for _ in range(n_queries)]

    for c_start in range(0, len(corpus_ids), corpus_chunk_size):
        c_end = min(c_start + corpus_chunk_size, len(corpus_ids))
        c_chunk = corpus_embs[c_start:c_end]
        scores = _score_chunk(q_prepared, c_chunk, similarity)  # (n_q, chunk_size)

        top_vals, top_idx = torch.topk(
            scores,
            min(max_k, c_end - c_start),
            dim=1,
            largest=True,
            sorted=False,
        )
        top_vals = top_vals.cpu().tolist()
        top_idx = top_idx.cpu().tolist()

        for qi in range(n_queries):
            for local_ci, score in zip(top_idx[qi], top_vals[qi]):
                cid = corpus_ids[c_start + local_ci]
                if len(heaps[qi]) < max_k:
                    heapq.heappush(heaps[qi], (score, cid))
                else:
                    heapq.heappushpop(heaps[qi], (score, cid))

    # 4. Per-query metrics
    per_query: list[QueryResult] = []
    agg_accumulator: dict[str, list] = {}

    for qi, qid in enumerate(query_ids):
        relevant = qrels.get(qid, set())
        # Queries with no relevant docs have undefined metrics (every helper
        # returns 0). Including them in the mean biases the aggregate toward 0,
        # so drop them from both per_query output and the aggregate accumulator.
        if not relevant:
            continue

        ranked = sorted(heaps[qi], key=lambda x: x[0], reverse=True)
        ranked_scores = [r[0] for r in ranked]
        ranked_ids = [r[1] for r in ranked]

        q_metrics = _compute_query_metrics(ranked_ids, relevant, ks)

        for k, v in q_metrics.items():
            agg_accumulator.setdefault(k, []).append(v)

        per_query.append(QueryResult(
            query_id=qid,
            query_text=query_texts[qi],
            relevant_ids=relevant,
            ranked_ids=ranked_ids,
            ranked_scores=ranked_scores,
            metrics=q_metrics,
        ))

    # 5. Aggregate
    aggregate = {metric: float(np.mean(vals)) for metric, vals in agg_accumulator.items()}

    return SubsetResult(
        dataset=dataset,
        subset=subset,
        per_query=per_query,
        aggregate=aggregate,
        emb_info=emb_info,
    )
