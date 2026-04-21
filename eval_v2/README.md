# eval_v2

Modular retrieval evaluation pipeline for embedding compression experiments.

Replaces the monolithic `eval/eval.py` with a clean, layered architecture that adds:
- **Per-query results** — required for statistical significance testing
- **Embedding cache for NanoBEIR** — shared across experiments with the same base checkpoint
- **All-metric significance testing** — Wilcoxon signed-rank + Cohen's d over every metric × subset pair
- **Interactive HTML dashboard** — Plotly.js, fully in-browser, no matplotlib charts generated
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
│   └── significance.py    # Wilcoxon signed-rank + Cohen's d, all metrics × all subsets
├── wrappers/
│   ├── base.py            # DummyModel, hamming_similarity, embedding_info
│   ├── transforms.py      # apply_transforms() — truncate / binarize / INT8/4 quantize
│   ├── sentence_transformer.py  # load_base_model() factory for cache population
│   └── _compat_wrappers.py      # Re-exports PQ / TurboQuant wrappers from eval/
├── results/
│   └── store.py           # Atomic JSON I/O — aggregate, per_query, emb_info, significance
├── html/
│   ├── builder.py         # Bundles all results into a self-contained HTML file
│   └── template.html      # Plotly.js interactive dashboard (5 views)
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
6. Write `eval_v2/outputs/results/<dataset>/explorer.html` — open in any browser.

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
│   │   ├── aggregate.json     # model → subset → metric@k → float
│   │   ├── per_query.json     # model → subset → query_id → {metrics, ranked_ids, relevant_ids}
│   │   ├── emb_info.json      # model → subset → {queries, corpus} → {n, dim, element_size_bit}
│   │   ├── significance.json  # "{modelA}|||{modelB}|||{subset}|||{metric}" → {p, effect, wins, ties, losses, n}
│   │   └── explorer.html      # Self-contained interactive dashboard
│   ├── beir/
│   │   └── ...
│   └── nasa_smd_ir/
│       └── ...
└── cache/
    └── <model_hash>/<dataset>/<subset>/
        ├── corpus.npz
        └── queries.npz
```

### Embedding cache

```
eval_v2_cache/
└── <model_hash>/
    └── <dataset>/<subset>/
        ├── corpus.npz   # float32, shape (N, D)
        └── queries.npz  # float32, shape (M, D)
```

The cache key is `sha256(model_path)[:16]`. Experiments that share the same base
checkpoint (e.g. all MRL variants trained from the same fine-tuned model) automatically
reuse the same cached embeddings — transforms are applied on load.

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

## Interactive HTML dashboard

Open `explorer.html` in any browser — no server needed.

| Tab | What you see |
|---|---|
| **Model Comparison** | Grouped bar chart — metric@k for the chosen subset, with hatch patterns |
| **Size vs Perf** | Scatter — embedding size (bits) vs performance, marker shapes from model config |
| **Query Browser** | Pick a subset and query → ranked results per model side-by-side, relevant docs highlighted |
| **Significance** | Heatmap of p-values (Wilcoxon) for the chosen metric — red = significant (p < 0.05) |
| **Metric@K Trend** | Line chart — how a metric changes as K varies, per model |

Use the **sidebar** to filter by subset, metric, K value, or model group. Use Plotly's
built-in camera icon to save any plot as PNG.

---

## Future extensions

The architecture is designed to support:

| Feature | What to add |
|---|---|
| **H-EQAT** (multi-level quant) | `quant_levels: list[int]` field in `ModelSpec`; cache stores `{bits: array}` dict |
| **Unified H-EQAT + MRL** | `mrl_dims: list[int]` in `ModelSpec`; cache key includes `(mrl_dim, bits)` |
| **Student-Teacher** | New entry in `MODELS` only |
| **Annealed Hard Tanh** | New wrapper in `wrappers/` + register in `ModelSpec` |
| **TurboQuant + MRL** | Compose `truncate_dim` + `tq_bits` in existing `ModelSpec` fields |
| **New dataset** | Add `DatasetSpec` to `config/datasets.py` + loader function in `run.py` |
