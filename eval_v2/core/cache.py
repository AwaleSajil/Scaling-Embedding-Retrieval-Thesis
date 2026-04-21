"""
Embedding cache for eval_v2.

Cache layout on disk:
    <root>/<model_hash>/<dataset>/<subset>/<split>.npz

where:
    model_hash  = sha256(model_path)[:16]
    split       = "corpus" | "queries"

For standard models, each .npz contains a single array "embeddings" (float32, N×D).
For future H-EQAT multi-level models, the .npz contains multiple arrays keyed by
bit-level, e.g. "level_32", "level_8", "level_4", "level_1".

Multi-GPU pool is opened once per (model_path, dataset, subset, split) call.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch

from eval_v2.wrappers.sentence_transformer import load_base_model
from eval_v2.config.models import ModelSpec


def _model_hash(model_path: str) -> str:
    return hashlib.sha256(model_path.encode()).hexdigest()[:16]


def _cache_path(root: str, model_path: str, dataset: str, subset: str, split: str) -> Path:
    h = _model_hash(model_path)
    subset_str = subset if subset else "no_subset"
    return Path(root) / h / dataset / subset_str / f"{split}.npz"


class EmbeddingCache:
    """
    Manages disk-cached raw float32 embeddings.

    Experiments that share the same base model_path automatically reuse
    each other's cache entry — transforms (truncation, binarization, quantization)
    are applied on load by wrappers/transforms.py, not here.
    """

    def __init__(self, root: str, batch_size: int = 32, chunk_size: int = 1000):
        self.root = root
        self.batch_size = batch_size
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_compute(
        self,
        model_path: str,
        texts: list[str],
        dataset: str,
        subset: str,
        split: str,           # "corpus" | "queries"
        spec: ModelSpec | None = None,
    ) -> np.ndarray:
        """
        Return raw float32 embeddings for *texts*.
        If already on disk, load and return.
        Otherwise encode with multi-GPU pool, save, and return.

        Returns ndarray of shape (N, D) with dtype float32.
        """
        path = _cache_path(self.root, model_path, dataset, subset, split)

        if path.exists():
            print(f"[Cache] HIT  {path}")
            data = np.load(path)
            return data["embeddings"]

        print(f"[Cache] MISS {path} — encoding {len(texts)} texts ...")
        embeddings = self._encode(model_path, texts, spec)

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, embeddings=embeddings)
        self._write_meta(path, model_path, dataset, subset, split, len(texts), embeddings.shape[1])

        print(f"[Cache] Saved  {path}  shape={embeddings.shape}")
        return embeddings

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _encode(self, model_path: str, texts: list[str], spec: ModelSpec | None) -> np.ndarray:
        """Encode *texts* with a fresh SentenceTransformer using multi-GPU pool."""
        # Load plain base model (no compression wrappers).
        # Pass spec only for pooling_mode customisation; compression done later.
        if spec is not None:
            model = load_base_model(spec)
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_path)

        pool = model.start_multi_process_pool()
        try:
            embeddings = model.encode(
                texts,
                pool=pool,
                batch_size=self.batch_size,
                chunk_size=self.chunk_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        finally:
            model.stop_multi_process_pool(pool)

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        return np.asarray(embeddings, dtype=np.float32)

    def _write_meta(
        self,
        npz_path: Path,
        model_path: str,
        dataset: str,
        subset: str,
        split: str,
        n: int,
        dim: int,
    ) -> None:
        meta_path = npz_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_path": model_path,
                    "dataset": dataset,
                    "subset": subset,
                    "split": split,
                    "n": n,
                    "dim": dim,
                },
                f,
                indent=2,
            )

    def invalidate(self, model_path: str, dataset: str, subset: str, split: str) -> None:
        path = _cache_path(self.root, model_path, dataset, subset, split)
        if path.exists():
            path.unlink()
            print(f"[Cache] Invalidated {path}")
