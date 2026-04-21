"""
Thin re-exports of the compression wrappers from the original eval/custum_evals.py.
These are only needed when PQ/TurboQuant fitting must happen at eval time.
For the common case (cosine, hamming, int8/4), transforms.py handles everything
without loading these wrappers.
"""
import sys
import os

# Point at the original eval directory so the imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../eval"))

from custum_evals import (   # noqa: F401
    PQSentenceTransformer,
    TurboQuantSentenceTransformer,
    QuantizedSentenceTransformer,
    UBinarySentenceTransformer,
)
