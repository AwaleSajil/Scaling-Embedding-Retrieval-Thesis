from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelSpec:
    path: str
    display_name: str
    group: str
    color: str
    hatch: str
    marker: str           # Plotly symbol name (circle, square, diamond, ...)
    similarity: str = "cosine"        # "cosine" or "hamming"
    truncate_dim: Optional[int] = None
    quant_bits: Optional[int] = None  # 8 or 4 → int8/int4
    pq_M: Optional[int] = None
    pq_nbits: int = 8
    tq_bits: Optional[int] = None
    pooling_mode: Optional[str] = None
    model_kwargs: dict = field(default_factory=dict)
    # Future H-EQAT / Unified Framework fields (uncomment when needed):
    # quant_levels: Optional[list[int]] = None   # e.g. [32, 8, 4, 1] bits per level
    # mrl_dims: Optional[list[int]] = None       # e.g. [32, 64, 128] for unified framework


# ---------------------------------------------------------------------------
# Model registry
# Each entry is keyed by a short experiment ID.
# ALL visual and algorithmic properties live here — nothing defined elsewhere.
# ---------------------------------------------------------------------------
MODELS: dict[str, ModelSpec] = {

    # --- Baselines (blue family, hatch: //, marker: circle) ---
    "bge-base-en-v1.5": ModelSpec(
        path="BAAI/bge-base-en-v1.5",
        display_name="BGE-base",
        group="Baseline",
        color="#1f77b4",
        hatch="//",
        marker="circle",
    ),
    "finetuned-bge-base-en-v1.5": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        display_name="FT",
        group="Baseline",
        color="#6baed6",
        hatch="//",
        marker="circle",
    ),

    # --- Post-quantization from finetuned (purple family, hatch: xx, marker: square) ---
    "finetuned-bge-base-en-v1.5-post-int8": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        display_name="FT + INT8",
        group="Post-Quant",
        color="#756bb1",
        hatch="xx",
        marker="square",
        quant_bits=8,
    ),
    "finetuned-bge-base-en-v1.5-post-binary": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        display_name="FT + Bin",
        group="Post-Quant",
        color="#bcbddc",
        hatch="xx",
        marker="square",
        similarity="hamming",
    ),

    # --- BAT (brown, hatch: --, marker: diamond) ---
    "finetuned-bge-base-en-v1.5-bat": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_21-07-12/bge-base-en-v1.5/final_model",
        display_name="BAT",
        group="BAT",
        color="#8c564b",
        hatch="--",
        marker="diamond",
        similarity="hamming",
    ),

    # --- MRL float (green gradient, lighter=smaller dim, hatch: ||, marker: triangle-up) ---
    "finetuned-bge-base-en-v1.5-mrl-32": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        display_name="MRL-32",
        group="MRL",
        color="#c7e9c0",
        hatch="||",
        marker="triangle-up",
        truncate_dim=32,
    ),
    "finetuned-bge-base-en-v1.5-mrl-64": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        display_name="MRL-64",
        group="MRL",
        color="#74c476",
        hatch="||",
        marker="triangle-up",
        truncate_dim=64,
    ),
    "finetuned-bge-base-en-v1.5-mrl-128": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        display_name="MRL-128",
        group="MRL",
        color="#31a354",
        hatch="||",
        marker="triangle-up",
        truncate_dim=128,
    ),
    "finetuned-bge-base-en-v1.5-mrl-256": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        display_name="MRL-256",
        group="MRL",
        color="#006d2c",
        hatch="||",
        marker="triangle-up",
        truncate_dim=256,
    ),
    "finetuned-bge-base-en-v1.5-mrl-512": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        display_name="MRL-512",
        group="MRL",
        color="#00441b",
        hatch="||",
        marker="triangle-up",
        truncate_dim=512,
    ),

    # --- MRL + post-binary (orange→red gradient, lighter=smaller dim, hatch: \\, marker: triangle-down) ---
    "finetuned-bge-base-en-v1.5-mrl-32-post-binary": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        display_name="MRL-32 + Bin",
        group="MRL+Binary",
        color="#fdd0a2",
        hatch="\\\\",
        marker="triangle-down",
        truncate_dim=32,
        similarity="hamming",
    ),
    "finetuned-bge-base-en-v1.5-mrl-64-post-binary": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        display_name="MRL-64 + Bin",
        group="MRL+Binary",
        color="#fdae6b",
        hatch="\\\\",
        marker="triangle-down",
        truncate_dim=64,
        similarity="hamming",
    ),
    "finetuned-bge-base-en-v1.5-mrl-128-post-binary": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        display_name="MRL-128 + Bin",
        group="MRL+Binary",
        color="#f16913",
        hatch="\\\\",
        marker="triangle-down",
        truncate_dim=128,
        similarity="hamming",
    ),
    "finetuned-bge-base-en-v1.5-mrl-256-post-binary": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        display_name="MRL-256 + Bin",
        group="MRL+Binary",
        color="#d94801",
        hatch="\\\\",
        marker="triangle-down",
        truncate_dim=256,
        similarity="hamming",
    ),
    "finetuned-bge-base-en-v1.5-mrl-512-post-binary": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        display_name="MRL-512 + Bin",
        group="MRL+Binary",
        color="#7f2704",
        hatch="\\\\",
        marker="triangle-down",
        truncate_dim=512,
        similarity="hamming",
    ),

    # --- PQ only, full dim=768 (rose/crimson, hatch: ++, marker: star) ---
    "finetuned-bge-base-en-v1.5-pq-m16-n8": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        display_name="PQ (M16, b8)",
        group="PQ",
        color="#c9184a",
        hatch="++",
        marker="star",
        pq_M=16,
        pq_nbits=8,
    ),
    "finetuned-bge-base-en-v1.5-pq-m16-n4": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        display_name="PQ (M16, b4)",
        group="PQ",
        color="#ff758f",
        hatch="++",
        marker="star",
        pq_M=16,
        pq_nbits=4,
    ),

    # --- MRL + PQ (navy→teal gradient, hatch: oo, marker: hexagon) ---
    "finetuned-bge-base-en-v1.5-mrl-512-pq-m16-n8": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        display_name="MRL-512 + PQ (M16, b8)",
        group="MRL+PQ",
        color="#03045e",
        hatch="oo",
        marker="hexagon",
        truncate_dim=512,
        pq_M=16,
        pq_nbits=8,
    ),
    "finetuned-bge-base-en-v1.5-mrl-256-pq-m16-n8": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        display_name="MRL-256 + PQ (M16, b8)",
        group="MRL+PQ",
        color="#023e8a",
        hatch="oo",
        marker="hexagon",
        truncate_dim=256,
        pq_M=16,
        pq_nbits=8,
    ),
    "finetuned-bge-base-en-v1.5-mrl-128-pq-m16-n8": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        display_name="MRL-128 + PQ (M16, b8)",
        group="MRL+PQ",
        color="#014f86",
        hatch="oo",
        marker="hexagon",
        truncate_dim=128,
        pq_M=16,
        pq_nbits=8,
    ),
    "finetuned-bge-base-en-v1.5-mrl-128-pq-m16-n4": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        display_name="MRL-128 + PQ (M16, b4)",
        group="MRL+PQ",
        color="#0077b6",
        hatch="oo",
        marker="hexagon",
        truncate_dim=128,
        pq_M=16,
        pq_nbits=4,
    ),
    "finetuned-bge-base-en-v1.5-mrl-64-pq-m8-n8": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        display_name="MRL-64 + PQ (M8, b8)",
        group="MRL+PQ",
        color="#0096c7",
        hatch="oo",
        marker="hexagon",
        truncate_dim=64,
        pq_M=8,
        pq_nbits=8,
    ),
    "finetuned-bge-base-en-v1.5-mrl-64-pq-m8-n4": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        display_name="MRL-64 + PQ (M8, b4)",
        group="MRL+PQ",
        color="#00b4d8",
        hatch="oo",
        marker="hexagon",
        truncate_dim=64,
        pq_M=8,
        pq_nbits=4,
    ),
    "finetuned-bge-base-en-v1.5-mrl-32-pq-m4-n8": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        display_name="MRL-32 + PQ (M4, b8)",
        group="MRL+PQ",
        color="#48cae4",
        hatch="oo",
        marker="hexagon",
        truncate_dim=32,
        pq_M=4,
        pq_nbits=8,
    ),
    "finetuned-bge-base-en-v1.5-mrl-32-pq-m4-n4": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260408_01-54-42/bge-base-en-v1.5/final_model",
        display_name="MRL-32 + PQ (M4, b4)",
        group="MRL+PQ",
        color="#90e0ef",
        hatch="oo",
        marker="hexagon",
        truncate_dim=32,
        pq_M=4,
        pq_nbits=4,
    ),

    # --- MRL + BAT (pink/magenta gradient, hatch: **, marker: cross) ---
    "finetuned-bge-base-en-v1.5-mrl-notrunc-bat": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260411_10-35-45/bge-base-en-v1.5/final_model",
        display_name="MRL-BAT",
        group="MRL+BAT",
        color="#7b0051",
        hatch="**",
        marker="cross",
        similarity="hamming",
    ),
    "finetuned-bge-base-en-v1.5-mrl-512-bat": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260411_10-35-45/bge-base-en-v1.5/final_model",
        display_name="MRL-512+BAT",
        group="MRL+BAT",
        color="#ad1457",
        hatch="**",
        marker="cross",
        truncate_dim=512,
        similarity="hamming",
    ),
    "finetuned-bge-base-en-v1.5-mrl-256-bat": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260411_10-35-45/bge-base-en-v1.5/final_model",
        display_name="MRL-256+BAT",
        group="MRL+BAT",
        color="#e91e8c",
        hatch="**",
        marker="cross",
        truncate_dim=256,
        similarity="hamming",
    ),
    "finetuned-bge-base-en-v1.5-mrl-128-bat": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260411_10-35-45/bge-base-en-v1.5/final_model",
        display_name="MRL-128+BAT",
        group="MRL+BAT",
        color="#f06292",
        hatch="**",
        marker="cross",
        truncate_dim=128,
        similarity="hamming",
    ),
    "finetuned-bge-base-en-v1.5-mrl-64-bat": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260411_10-35-45/bge-base-en-v1.5/final_model",
        display_name="MRL-64+BAT",
        group="MRL+BAT",
        color="#f8bbd0",
        hatch="**",
        marker="cross",
        truncate_dim=64,
        similarity="hamming",
    ),
    "finetuned-bge-base-en-v1.5-mrl-32-bat": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20260411_10-35-45/bge-base-en-v1.5/final_model",
        display_name="MRL-32+BAT",
        group="MRL+BAT",
        color="#fce4ec",
        hatch="**",
        marker="cross",
        truncate_dim=32,
        similarity="hamming",
    ),

    # --- TurboQuant post-hoc (gold/amber gradient, hatch: OO, marker: pentagon) ---
    "finetuned-bge-base-en-v1.5-tq-8bit": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        display_name="FT + TQ-8b",
        group="TurboQuant",
        color="#7b4f00",
        hatch="OO",
        marker="pentagon",
        tq_bits=8,
    ),
    "finetuned-bge-base-en-v1.5-tq-4bit": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        display_name="FT + TQ-4b",
        group="TurboQuant",
        color="#d4a017",
        hatch="OO",
        marker="pentagon",
        tq_bits=4,
    ),
    "finetuned-bge-base-en-v1.5-tq-3bit": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        display_name="FT + TQ-3b",
        group="TurboQuant",
        color="#f0c040",
        hatch="OO",
        marker="pentagon",
        tq_bits=3,
    ),
    "finetuned-bge-base-en-v1.5-tq-2bit": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        display_name="FT + TQ-2b",
        group="TurboQuant",
        color="#fce48a",
        hatch="OO",
        marker="pentagon",
        tq_bits=2,
    ),
    "finetuned-bge-base-en-v1.5-tq-1bit": ModelSpec(
        path="/rhome/sawale/thesis/models/nrows_None__nsrc_None/timestamp_20251213_13-26-29/bge-base-en-v1.5/final_model",
        display_name="FT + TQ-1b",
        group="TurboQuant",
        color="#fff5cc",
        hatch="OO",
        marker="pentagon",
        tq_bits=1,
    ),
}

# Ordered list of all model keys (preserves insertion order for display)
MODEL_KEYS: list[str] = list(MODELS.keys())

# All unique groups in display order
MODEL_GROUPS: list[str] = list(dict.fromkeys(s.group for s in MODELS.values()))
