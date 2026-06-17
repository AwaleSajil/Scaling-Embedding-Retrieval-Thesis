"""
HTML dashboard builder.

Reads all result JSONs from a ResultsStore and injects them as a single
JSON payload into the Plotly.js template, producing a fully self-contained
HTML file (Plotly.js embedded inline — no internet required).
"""
import json
import math
import os
from pathlib import Path

from eval_v2.config.models import MODELS, ModelSpec
from eval_v2.results.store import ResultsStore


def _round_floats(obj, ndigits: int = 4):
    """Recursively round all floats to *ndigits* decimal places.

    Keeps ints, strings, and bools unchanged.
    Non-finite floats (NaN / ±inf) are replaced with None so they become
    valid JSON ``null`` — JavaScript's ``JSON.parse`` rejects bare ``NaN``.
    """
    if isinstance(obj, float):
        return round(obj, ndigits) if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def _safe_json_for_script(payload: dict) -> str:
    """
    Serialize JSON for embedding inside a <script> tag safely.
    Escapes closing tags so query text like '</script>' cannot terminate
    the script block and break page parsing.
    """
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


# Matplotlib hatch → Plotly fillpattern shape
_HATCH_TO_PLOTLY = {
    "//": "/",
    "\\\\": "\\",
    "xx": "x",
    "--": "-",
    "||": "|",
    "++": "+",
    "+-": "+",
    "oo": ".",
    "**": "x",
    "OO": ".",
    "~~": "-",
}


def _models_meta(models: dict[str, ModelSpec]) -> dict:
    return {
        key: {
            "display_name": spec.display_name,
            "group": spec.group,
            "base_model": spec.base_model,
            "color": spec.color,
            "marker": spec.marker,
            "hatch": spec.hatch,
            "plotly_pattern": _HATCH_TO_PLOTLY.get(spec.hatch, ""),
            "n_step": spec.n_step,
        }
        for key, spec in models.items()
    }


def build_html(store: ResultsStore, output_path: str) -> None:
    """Build a self-contained HTML dashboard and write it to *output_path*."""
    aggregate = store.load_aggregate()
    per_query = store.load_per_query()
    emb_info = store.load_emb_info()
    significance = store.load_significance()

    all_subsets: set[str] = set()
    for model_data in aggregate.values():
        all_subsets.update(model_data.keys())
    subsets = sorted(s for s in all_subsets if s != "mean")
    if "mean" in all_subsets:
        subsets.append("mean")

    query_texts: dict[str, dict[str, str]] = {}
    for model_data in per_query.values():
        for subset, queries in model_data.items():
            if subset not in query_texts:
                query_texts[subset] = {}
            for qid, qdata in queries.items():
                if qid not in query_texts[subset]:
                    query_texts[subset][qid] = qdata.get("text", "")
        if all(subset in query_texts for subset in subsets):
            break

    per_query_slim: dict = {}
    for model_key, model_data in per_query.items():
        per_query_slim[model_key] = {}
        for subset, queries in model_data.items():
            per_query_slim[model_key][subset] = {}
            for qid, qdata in queries.items():
                per_query_slim[model_key][subset][qid] = {
                    k: v for k, v in qdata.items() if k not in ("text",)
                }

    # Precompute per-(model, subset, metric@k) std and n so the dashboard
    # can render error bars without a fetch() call (which fails on file://).
    stats: dict = {}
    for model_key, model_data in per_query_slim.items():
        stats[model_key] = {}
        for subset, queries in model_data.items():
            metric_samples: dict[str, list] = {}
            for qdata in queries.values():
                for mk, v in qdata.items():
                    if "@" in mk and isinstance(v, (int, float)) and math.isfinite(v):
                        metric_samples.setdefault(mk, []).append(v)
            stats[model_key][subset] = {}
            for mk, samples in metric_samples.items():
                n = len(samples)
                mean = sum(samples) / n
                std = math.sqrt(sum((x - mean) ** 2 for x in samples) / n)
                stats[model_key][subset][mk] = {"std": std, "n": n}

    payload = {
        "models_meta": _models_meta(MODELS),
        "model_keys": [k for k in MODELS if k in aggregate],
        "subsets": subsets,
        "aggregate": _round_floats(aggregate),
        "emb_info": emb_info,
        "query_texts": query_texts,
        "stats": _round_floats(stats),
        # significance and per_query are lazy-loaded from companion shard files
        "significance": None,
        "per_query": None,
    }

    template_path = Path(__file__).parent / "template.html"
    with open(template_path, encoding="utf-8") as f:
        template = f.read()

    html = template.replace("__DATA_JSON__", _safe_json_for_script(payload))

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[HTML] Dashboard written → {output_path}")

    # Write per-(subset, metric) significance shards — fetched on demand by the UI
    sig_dir = os.path.join(out_dir, "significance_shards")
    os.makedirs(sig_dir, exist_ok=True)
    sig_shards: dict[str, dict] = {}
    for key, val in _round_floats(significance).items():
        parts = key.split("|||")
        if len(parts) != 4:
            continue
        _, _, subset_key, metric_key = parts
        shard_key = f"{subset_key}|||{metric_key}"
        if shard_key not in sig_shards:
            sig_shards[shard_key] = {}
        sig_shards[shard_key][key] = val
    for shard_key, shard_data in sig_shards.items():
        subset_key, metric_key = shard_key.split("|||")
        fname = f"{subset_key}__{metric_key.replace('@', '-')}.json"
        with open(os.path.join(sig_dir, fname), "w", encoding="utf-8") as f:
            json.dump(shard_data, f, ensure_ascii=False)
    print(f"[HTML] Significance shards → {sig_dir}/ ({len(sig_shards)} files)")

    # Write per-subset per_query shards — fetched on demand by the UI
    pq_dir = os.path.join(out_dir, "per_query_shards")
    os.makedirs(pq_dir, exist_ok=True)
    pq_rounded = _round_floats(per_query_slim)
    pq_shards: dict[str, dict] = {}
    for model_key, model_data in pq_rounded.items():
        for subset_key, queries in model_data.items():
            if subset_key not in pq_shards:
                pq_shards[subset_key] = {}
            pq_shards[subset_key][model_key] = queries
    for subset_key, shard_data in pq_shards.items():
        fname = f"{subset_key}.json"
        with open(os.path.join(pq_dir, fname), "w", encoding="utf-8") as f:
            json.dump(shard_data, f, ensure_ascii=False)
    print(f"[HTML] Per-query shards   → {pq_dir}/ ({len(pq_shards)} files)")
