"""
Shared utilities and the DummyModel stub used when corpus/query embeddings
are already pre-computed (no actual encoding needed at eval time).
"""
import numpy as np
import torch
from torch import Tensor


class _DummyModelCardData:
    def set_evaluation_metrics(self, *args, **kwargs):
        pass


class DummyModel:
    """
    Stub that mimics a SentenceTransformer for the ST evaluator API.
    Used when embeddings are sourced from cache — no GPU work at eval time.
    """
    def __init__(self, similarity_fn_name: str = "cosine"):
        from sentence_transformers import util as st_util
        self.similarity_fn_name = similarity_fn_name
        self.similarity = st_util.cos_sim
        self.model_card_data = _DummyModelCardData()

    def start_multi_process_pool(self, *args, **kwargs) -> dict:
        return {}

    def stop_multi_process_pool(self, pool: dict) -> None:
        pass


def embedding_dtype_bits(emb) -> int:
    """Return the storage width in bits of a single element in *emb*."""
    if isinstance(emb, torch.Tensor):
        dtype_str = str(emb.dtype)
    else:
        arr = np.asarray(emb)
        dtype_str = str(arr.dtype)

    _map = {
        "float32": 32, "torch.float32": 32, "float": 32,
        "float16": 16, "torch.float16": 16,
        "bfloat16": 16, "torch.bfloat16": 16,
        "uint8": 8,  "torch.uint8": 8,
        "int8": 8,   "torch.int8": 8,
    }
    return _map.get(dtype_str, 32)


def embedding_info(emb) -> dict:
    """Return {n, dimension, element_size_bit} for an embedding array/tensor."""
    if isinstance(emb, torch.Tensor):
        arr = emb.detach().cpu()
    else:
        arr = np.asarray(emb)
    n = int(arr.shape[0])
    dim = int(arr.shape[1])
    bits = embedding_dtype_bits(emb)
    return {"n": n, "dimension": dim, "element_size_bit": bits}


def hamming_similarity(a: Tensor, b: Tensor) -> Tensor:
    """
    Memory-efficient Hamming similarity for packed ubinary tensors (uint8).
    Returns a float32 (n_queries, n_corpus) matrix where higher = more similar.
    """
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)

    a = a.cpu().to(torch.uint8)
    b = b.cpu().to(torch.uint8)

    n_q, n_c = a.shape[0], b.shape[0]
    vec_bits = a.shape[1] * 8
    chunk = min(500, n_c)
    result = torch.zeros((n_q, n_c), dtype=torch.float32)
    lut = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.int32)

    for start in range(0, n_c, chunk):
        end = min(start + chunk, n_c)
        diff = a.unsqueeze(1) ^ b[start:end].unsqueeze(0)
        counts = lut[diff.view(-1).long()].view(n_q, end - start, a.shape[1])
        result[:, start:end] = (vec_bits - counts.sum(dim=2).float())

    return result
