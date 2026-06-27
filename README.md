# Compact Embeddings for Information Retrieval

Master's thesis code: fine-tuning sentence embedding models with binarization and Matryoshka Representation Learning (MRL) for efficient retrieval. The repo contains the training pipeline (`src/`) and the evaluation pipeline (`eval_v2/`), which are independent and only share the model registry in `eval_v2/config/models.py`.

## Setup

This project uses [uv](https://docs.astral.sh/uv/). Install it, then sync the environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
uv sync                                            # creates .venv and installs pinned deps
```

This reads `pyproject.toml` and `uv.lock` and pins Python 3.11 (`.python-version`). Run commands inside the environment with `uv run`, e.g. `uv run python -m eval_v2.run`.

> On Apple Silicon, `uv sync` installs the CPU/MPS build of torch. The multi-GPU `torchrun` training below requires a CUDA host.

Alternatively, install into an existing environment with `pip install -r requirements.txt`.

Then copy `.env.example` to `.env` and fill in the credentials used at training time:

```bash
cp .env.example .env
```

```
WANDB_API_KEY=...
HUGGINGFACE_TOKEN=...
```

## Training

All training runs go through `src/train.py`. Defaults come from `src/config.yaml`, CLI flags override them, and `--expirement_number` selects the experiment variant.

```bash
cd experiments
torchrun --nproc_per_node=4 ../src/train.py --expirement_number e5
# override config values inline:
torchrun --nproc_per_node=4 ../src/train.py --expirement_number e5 --lr 1e-5 --batch_size 64
```

Key CLI flags: `--expirement_number`, `--nrows`, `--n_data_src`, `--model_name`, `--batch_size`, `--num_train_epochs`, `--lr`, `--gradient_accumulation_steps`, `--resume_checkpoint_path`, `--resume_run_id`.

The base encoder is `BAAI/bge-base-en-v1.5`; experiments also support `FacebookAI/roberta-base` and `microsoft/mpnet-base` via `--model_name`. Training uses 8 mixed retrieval datasets with weighted `MultipleNegativesRankingLoss`, runs multi-GPU via DDP (`torchrun`), and writes the final checkpoint to `models/nrows_None__nsrc_None/timestamp_<ts>/<model>/final_model/`.

### Experiment variants

| Number   | Description                                            |
|----------|--------------------------------------------------------|
| `e1`     | Baseline fine-tune (cosine, MNRL loss)                 |
| `e2`     | Baseline + STE binarization layer                      |
| `e3`     | MRL training (Matryoshka over `[32,64,128,256,512]`)   |
| `e4`     | MRL + STE binarization                                  |
| `e5`     | Annealed tanh binarization                              |
| `e6_1b`–`e6_4b` | Multi-bit annealed sigmoid quantization (1–4 bit) |
| `e7`     | MRL + annealed tanh binarization                       |

Convenience wrappers for each experiment live alongside the training entry point as [experiments/e1_run_baseline.sh](experiments/e1_run_baseline.sh) through [experiments/e7_run_atanh_mrl.sh](experiments/e7_run_atanh_mrl.sh).

### Binarization layers

[src/utils/utils.py](src/utils/utils.py) defines:
- `BinarizationLayer`: Straight-Through Estimator, `sign(x)` forward, identity backward.
- `AnnealedTanhBinarizationLayer`: training uses `tanh(β·x)` where `β = sqrt(γ·step + 1)` (β grows from 1 → ~28 over 8k steps); inference uses `sign(x)`. `BetaAnnealCallback` in [src/train.py](src/train.py) calls `.anneal_step()` after every optimizer step and logs β to W&B.

## Evaluation

```bash
# Evaluate all registered models on NanoBEIR
python -m eval_v2.run

# Evaluate a subset of models
python -m eval_v2.run --models bge-base-en-v1.5 finetuned-bge-base-en-v1.5-bat

# Rebuild the HTML dashboard without re-running eval
python -m eval_v2.run --just_html

# Common optional flags
python -m eval_v2.run --dataset beir --cache_dir eval_v2/outputs/cache/ --output_dir eval_v2/outputs/results/
```

### How evaluation works

- **Embedding cache** ([eval_v2/core/cache.py](eval_v2/core/cache.py)): Raw float32 embeddings are encoded once and stored at `eval_v2/outputs/cache/<sha256(model_path)[:16]>/nanobeir/<Subset>/{corpus,queries}.{npz,json}`. Multiple `ModelSpec` entries that share the same checkpoint path reuse the same cache. A cache miss triggers multi-GPU encoding.
- **Transform pipeline** ([eval_v2/wrappers/transforms.py](eval_v2/wrappers/transforms.py)): Applied at load time, after the cache, in this order:
  1. Truncate dimensions (`truncate_dim`)
  2. Binarize → uint8 packed (`similarity == "hamming"`)
  3. INT8/INT4 quantize (`quant_bits`)
  4. PQ reconstruct, fit on corpus, asymmetric (`pq_M`)
  5. TurboQuant dequantize, fit on corpus (`tq_bits`)
- **Model registry** ([eval_v2/config/models.py](eval_v2/config/models.py)): `MODELS: dict[str, ModelSpec]` is the single source of truth for every evaluated variant. `ModelSpec` fields control which transforms are applied; there is no separate config file. Post-hoc transforms (INT8, PQ, TQ, MRL truncations) of the same checkpoint share one cache entry and differ only in their `ModelSpec`.
- **Evaluator** ([eval_v2/core/evaluator.py](eval_v2/core/evaluator.py)): Scores corpus/query pairs using cosine or Hamming similarity, computes NDCG@k, MRR@k, Recall@k for k ∈ {1, 3, 5, 10}, runs paired significance tests across model pairs, and emits `SubsetResult` objects.

## Adding a new experiment

1. **Train**: add an experiment branch in [src/train.py](src/train.py) (follow the e5 pattern) and run with the new `--expirement_number`.
2. **Register**: add one or more `ModelSpec` entries to `MODELS` in [eval_v2/config/models.py](eval_v2/config/models.py) pointing to the new checkpoint path. If the experiment is a post-hoc transform of an existing checkpoint, no new training is needed; just a new spec.
3. **Evaluate**: `python -m eval_v2.run --models your_new_key`.

## Repository layout

```
src/                      Training pipeline (train.py, config.yaml, utils/)
experiments/              Per-experiment shell wrappers and notes
eval_v2/                  Evaluation pipeline (run.py, core/, wrappers/, config/)
  config/models.py        Model registry (ModelSpec entries)
  outputs/                Cache, results, dashboards
models/                   Saved checkpoints
data/                     Training data
docs/report/              Thesis report (LaTeX sources)
requirements.txt
.env                      WANDB_API_KEY, HUGGINGFACE_TOKEN
```
