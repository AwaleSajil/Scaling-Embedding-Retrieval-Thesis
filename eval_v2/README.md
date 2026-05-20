# eval_v2

Modular retrieval evaluation pipeline for embedding compression experiments.

Replaces the monolithic `eval/eval.py` with a clean, layered architecture that adds:
- **Per-query results** — required for statistical significance testing
- **Embedding cache for NanoBEIR** — shared across experiments with the same base checkpoint
- **All-metric significance testing** — Wilcoxon signed-rank + paired t-test + Cohen's d over every metric × subset pair
- **Interactive HTML dashboard** — Plotly.js, fully in-browser, lazy-loads heavy data on demand
- **Single model registry** — color, hatch, and marker all defined in one place

---

## Directory layout

```
eval_v2/
├── config/
│   ├── models.py          # ModelSpec dataclass + MODELS dict (all visual + algorithmic config)
│   └── datasets.py        # DatasetSpec + DATASETS dict
├── core/
│   ├── cache.py           # EmbeddingCache — multi-GPU encoding, shared across experiments
│   ├── evaluator.py       # run_ir_eval() — per-query NDCG, MRR, Accuracy, Recall
│   └── significance.py    # Wilcoxon + paired t-test + Cohen's d, all metrics × all subsets
├── wrappers/
│   ├── base.py            # DummyModel, hamming_similarity, embedding_info
│   ├── transforms.py      # apply_transforms() — truncate / binarize / INT8/4 quantize
│   ├── sentence_transformer.py  # load_base_model() factory for cache population
│   └── _compat_wrappers.py      # Re-exports PQ / TurboQuant wrappers from eval/
├── results/
│   └── store.py           # Atomic JSON I/O — aggregate, per_query, emb_info, significance
├── html/
│   ├── builder.py         # Writes explorer.html + companion shard files
│   └── template.html      # Plotly.js interactive dashboard (5 views, lazy data loading)
└── run.py                 # CLI entry point
```

---

## Quick start

```bash
cd /nas/rhome/sawale/thesis

python -m eval_v2.run \
  --dataset nanobeir \
  --batch_size 32
```

This will:
1. For each NanoBEIR subset, download corpus / queries / qrels from HuggingFace.
2. Encode and cache raw float32 embeddings once per unique checkpoint (multi-GPU).
3. Apply model-specific transforms (truncation, binarization, quantization, PQ, TurboQuant).
4. Run per-query IR evaluation and save results incrementally.
5. Compute pairwise significance tests for every metric × subset combination.
6. Write `eval_v2/outputs/results/<dataset>/explorer.html` and companion shard files.

### Useful flags

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `nanobeir` | Dataset to evaluate (`nanobeir`, `beir`, ...) |
| `--output_dir` | `eval_v2/outputs/results/` | Where JSON results are written |
| `--cache_dir` | `eval_v2/outputs/cache/` | Where raw float32 embeddings are cached |
| `--html_path` | `<output_dir>/<dataset>/explorer.html` | Output HTML path |
| `--batch_size` | `32` | Encoding batch size |
| `--ks` | `1 3 5 10` | K values for IR metrics |
| `--models` | all | Whitelist specific model keys |
| `--subsets` | all | Whitelist specific dataset subsets |
| `--just_html` | off | Rebuild HTML from existing results without re-evaluating |

### Resume / skip already-evaluated models

Results are saved after each (model, subset) pair. If the run is interrupted, re-running
the same command will skip already-completed pairs automatically.

### Rebuild the dashboard without re-evaluating

```bash
python -m eval_v2.run --just_html
```

---

## Output files

Each dataset gets its own subdirectory so runs never collide:

```
eval_v2/outputs/
├── results/
│   ├── nanobeir/
│   │   ├── aggregate.json          # model → subset → metric@k → float
│   │   ├── per_query.json          # model → subset → query_id → {metrics, ranked_ids, relevant_ids}
│   │   ├── emb_info.json           # model → subset → {queries, corpus} → {n, dim, element_size_bit}
│   │   ├── significance.json       # "{modelA}|||{modelB}|||{subset}|||{metric}" → {p, effect, wins, ...}
│   │   ├── explorer.html           # Lightweight dashboard (~2 MB), fetches shards on demand
│   │   ├── significance_shards/    # One JSON per (subset, metric) — fetched when Significance tab opens
│   │   │   ├── NanoArguAna__mrr-10.json
│   │   │   └── ...                 # ~208 files, ~3 MB each
│   │   └── per_query_shards/       # One JSON per subset — fetched when Query tab opens
│   │       ├── NanoArguAna.json
│   │       └── ...                 # 13 files, ~8 MB each
│   └── beir/
│       └── ...
└── cache/
    └── <model_hash>/<dataset>/<subset>/
        ├── corpus.npz
        └── queries.npz
```

### Embedding cache

The cache key is `sha256(model_path)[:16]`. Experiments that share the same base
checkpoint (e.g. all MRL variants trained from the same fine-tuned model) automatically
reuse the same cached embeddings — transforms are applied on load.

---

## Viewing the dashboard

The dashboard uses `fetch()` to load shard files on demand, so it must be served over HTTP
(browsers block `fetch()` for `file://` URLs).

```bash
cd eval_v2/outputs/results/nanobeir
python -m http.server 8080
# open http://localhost:8080/explorer.html
```

If you are on a remote server via VS Code SSH, the port is forwarded automatically and
opens in your local browser.

The initial page load is ~2 MB. Heavy data loads lazily:
- **Significance tab** — fetches one ~3 MB shard for the selected subset + metric; cached for the rest of the session.
- **Query tab** — fetches one ~8 MB shard for the selected subset; cached for the rest of the session.
- All other tabs (bar, scatter, table, line) use only the inline aggregate data and are instant.

---

## Interactive HTML dashboard

| Tab | What you see |
|---|---|
| **Model Comparison** | Grouped bar chart — metric@k for the chosen subset, with hatch patterns |
| **Size vs Perf** | Scatter — embedding size (MB) vs performance; std error bars load with per-query shard |
| **Metrics Table** | Sortable table of all metric@k values across models |
| **Query Browser** | Pick a subset and query → ranked results per model side-by-side, relevant docs highlighted |
| **Significance** | Heatmap of p-values (Wilcoxon / paired t-test) for the chosen subset + metric |
| **Metric@K Trend** | Line chart — how a metric changes as K varies, per model |

Use the **sidebar** to filter by subset, metric, K value, or model group. Click any significance
cell to open a normality diagnostics panel (Q-Q plot, histogram, Shapiro-Wilk test). Use
Plotly's built-in camera icon to save any plot as PNG.

---

## Adding a new model

Add one entry to `config/models.py`. Everything else (caching, evaluation, HTML, significance) picks it up automatically.

```python
MODELS["my_new_model"] = ModelSpec(
    path="/path/to/checkpoint",
    display_name="My Model",
    group="MyGroup",
    color="#e74c3c",
    hatch="//",
    marker="diamond",
    truncate_dim=128,          # optional
    similarity="cosine",       # "cosine" or "hamming"
    quant_bits=8,              # optional: 8 or 4
    # pq_M=16, pq_nbits=8,    # optional: Product Quantization
    # tq_bits=4,               # optional: TurboQuant
)
```

---

## Future extensions

| Feature | What to add |
|---|---|
| **H-EQAT** (multi-level quant) | `quant_levels: list[int]` field in `ModelSpec`; cache stores `{bits: array}` dict |
| **Unified H-EQAT + MRL** | `mrl_dims: list[int]` in `ModelSpec`; cache key includes `(mrl_dim, bits)` |
| **Student-Teacher** | New entry in `MODELS` only |
| **New dataset** | Add `DatasetSpec` to `config/datasets.py` + loader function in `run.py` |
