"""
Factory that loads the correct SentenceTransformer wrapper for a given ModelSpec.
Only used during cache population (encoding corpus / queries).
The wrapper type determines how .encode() behaves; all multi-GPU pool calls
are issued by the cache layer, not here.
"""
import sys
import os

from sentence_transformers import SentenceTransformer
from sentence_transformers import models as s_models

from eval_v2.config.models import ModelSpec

# Make the training src available for any custom modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


def load_base_model(spec: ModelSpec) -> SentenceTransformer:
    """
    Load a plain SentenceTransformer (no compression wrappers).
    Used for cache population: encode raw float32 embeddings once,
    then apply transforms (truncate / binarize / quantize) on load.
    """
    kwargs = spec.model_kwargs or {}
    truncate_dim = spec.truncate_dim

    if spec.pooling_mode:
        transformer = s_models.Transformer(spec.path)
        pooling = s_models.Pooling(
            word_embedding_dimension=transformer.get_word_embedding_dimension(),
            pooling_mode=spec.pooling_mode,
        )
        return SentenceTransformer(modules=[transformer, pooling], truncate_dim=truncate_dim)

    return SentenceTransformer(spec.path, truncate_dim=None, model_kwargs=kwargs)


def load_pq_model(spec: ModelSpec) -> "PQSentenceTransformer":
    """Load a PQ wrapper (used only when PQ fit is required at eval time)."""
    from eval_v2.wrappers._compat_wrappers import PQSentenceTransformer
    kwargs = spec.model_kwargs or {}
    return PQSentenceTransformer(
        spec.path,
        M=spec.pq_M,
        nbits=spec.pq_nbits,
        truncate_dim=spec.truncate_dim,
        model_kwargs=kwargs,
    )


def load_tq_model(spec: ModelSpec) -> "TurboQuantSentenceTransformer":
    from eval_v2.wrappers._compat_wrappers import TurboQuantSentenceTransformer
    kwargs = spec.model_kwargs or {}
    return TurboQuantSentenceTransformer(
        spec.path,
        tq_bits=spec.tq_bits,
        truncate_dim=spec.truncate_dim,
        model_kwargs=kwargs,
    )
