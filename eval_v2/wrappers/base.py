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


_BIT_SHIFTS = torch.arange(8, dtype=torch.uint8)


def _unpack_pm1(packed: Tensor, dtype=torch.float32) -> Tensor:
    """Packed uint8 bits -> dense {-1,+1} matrix of shape (n, n_bytes*8)."""
    shifts = _BIT_SHIFTS.to(packed.device)
    bits = (packed.unsqueeze(-1) >> shifts) & 1           # (n, n_bytes, 8)
    return bits.reshape(packed.shape[0], -1).to(dtype) * 2 - 1


def hamming_similarity(a: Tensor, b: Tensor, sub_chunk_bytes: int = 256_000_000) -> Tensor:
    """
    Hamming similarity for packed ubinary tensors (uint8).
    Returns a float32 (n_queries, n_corpus) matrix where higher = more similar,
    equal to vec_bits - hamming_distance.

    Computed as a matmul rather than by counting set bits. Unpacking to {-1,+1}
    gives dot = (#match - #mismatch) = vec_bits - 2*distance, so

        similarity = vec_bits - distance = (vec_bits + dot) / 2

    which is exact, not an approximation -- the result is bit-identical to the
    previous XOR + popcount-lookup implementation.

    Why it matters: the old version materialised (n_q, chunk, n_bytes) int64 for
    the lookup gather, which at BEIR query counts is ~2.7 GB of scratch per
    chunk and ran 111x slower per pair than cosine. Since 66 of the 224 model
    specs use hamming, that one function dominated the whole scoring phase --
    ~36 days of the projected runtime. As a matmul it costs about the same as
    cosine and reuses the same BLAS (or GPU) path.

    Device-preserving: unlike the previous implementation this does not force
    inputs to CPU, so it works unchanged if the caller moves tensors to a GPU.
    """
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)

    a = a.to(torch.uint8)
    b = b.to(torch.uint8)

    n_q, n_c = a.shape[0], b.shape[0]
    vec_bits = a.shape[1] * 8

    a_pm1 = _unpack_pm1(a)                                 # (n_q, vec_bits)

    # Only the corpus side is chunked, and only to bound the unpacked copy:
    # (chunk, vec_bits) float32. The caller already sizes its own chunks so the
    # (n_q, chunk) score matrix fits memory.
    per_row = vec_bits * 4
    sub = max(1, min(n_c, sub_chunk_bytes // max(per_row, 1)))

    result = torch.empty((n_q, n_c), dtype=torch.float32, device=a.device)
    for start in range(0, n_c, sub):
        end = min(start + sub, n_c)
        b_pm1 = _unpack_pm1(b[start:end])
        result[:, start:end] = (vec_bits + a_pm1 @ b_pm1.t()) / 2

    return result
