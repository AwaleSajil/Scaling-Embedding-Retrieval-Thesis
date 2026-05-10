"""
Stateless + calibrated post-processing transforms applied to raw float32 embeddings.
All transforms operate on numpy arrays (float32) and return numpy or torch tensors.

Order of operations (same as the original eval.py apply_model_transformations):
  1. Truncate dimensions          (truncate_dim)
  2. Binarize → ubinary           (similarity == "hamming")
  3. INT8/4 quantize              (quant_bits in {8, 4})
  4. PQ reconstruct               (pq_M set)  — handled in evaluator, needs corpus fit
  5. TurboQuant quant+dequant     (tq_bits set) — handled in evaluator, lazy-init
  6. Plain float passthrough      (everything else)
"""
import numpy as np
import torch
from sentence_transformers.quantization import quantize_embeddings

from eval_v2.config.models import ModelSpec


def apply_transforms(
    raw: np.ndarray,
    spec: ModelSpec,
    calibration_embeddings: np.ndarray | None = None,
) -> np.ndarray | torch.Tensor:
    """
    Apply model-specific transforms to raw float32 full-dim embeddings.
    PQ and TurboQuant are *not* applied here (they require corpus-level fitting
    or lazy init); the evaluator handles them after calling this function.

    For INT8/INT4 quantization, *calibration_embeddings* should be a representative
    sample (e.g. first 1000 corpus rows, already truncated if truncate_dim is set).
    Using the same calibration for both corpus and queries ensures they are quantized
    to the same per-dimension scale.
    """
    embs: np.ndarray = np.asarray(raw, dtype=np.float32)

    # 1. Truncate
    if spec.truncate_dim is not None:
        embs = embs[:, : spec.truncate_dim]

    # 2. Binarize (hamming similarity → ubinary packed bits)
    if spec.similarity == "hamming":
        result = quantize_embeddings(embs, precision="ubinary")
        return torch.from_numpy(np.asarray(result)) if not isinstance(result, torch.Tensor) else result

    # 3. INT8 / INT4 quantization
    if spec.quant_bits in (8, 4):
        precision = {8: "int8", 4: "int4"}[spec.quant_bits]
        if calibration_embeddings is None:
            calibration_embeddings = embs[: min(1000, len(embs))]
        result = quantize_embeddings(embs, precision=precision, calibration_embeddings=calibration_embeddings)
        return torch.from_numpy(np.asarray(result)) if not isinstance(result, torch.Tensor) else result

    # 4-6. PQ / TurboQuant / plain float — return torch tensor, evaluator does the rest
    return torch.from_numpy(embs.copy())
