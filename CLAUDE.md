# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Training

Run from the `experiments/` directory:
```bash
cd experiments
torchrun --nproc_per_node=4 ../src/train.py --expirement_number e5
# override config values inline:
torchrun --nproc_per_node=4 ../src/train.py --expirement_number e5 --lr 1e-5 --batch_size 64
```

Key CLI flags: `--expirement_number`, `--nrows`, `--n_data_src`, `--model_name`, `--batch_size`, `--num_train_epochs`, `--lr`, `--resume_checkpoint_path`.

Credentials (`WANDB_API_KEY`, `HUGGINGFACE_TOKEN`) are loaded from `.env` via python-dotenv.

## Evaluation

```bash
# Evaluate all registered models on NanoBEIR
python -m eval_v2.run

# Evaluate a subset of models
python -m eval_v2.run --models bge-base-en-v1.5 finetuned-bge-base-en-v1.5-bat

# Rebuild the HTML dashboard without re-running eval
python -m eval_v2.run --just_html

# Key optional flags
python -m eval_v2.run --dataset beir --cache_dir eval_v2/outputs/cache/ --output_dir eval_v2/outputs/results/
```

## Architecture

The repo has two independent systems that share only `eval_v2/config/models.py`.

### Training system (`src/`)

`src/train.py` is the single entry point. It loads `src/config.yaml` as defaults, overrides with CLI args, then selects an experiment variant:

| Number | Experiment |
|--------|-----------|
| e1 | Baseline fine-tune (cosine, MNRL loss) |
| e2 | + STE binarization layer |
| e3 | MRL training (MatryoshkaLoss over `[32,64,128,256,512]` dims) |
| e4 | MRL + STE binarization |
| e5 | Annealed tanh binarization |

All experiments fine-tune `BAAI/bge-base-en-v1.5` on 8 mixed retrieval datasets with weighted `MultipleNegativesRankingLoss`. Multi-GPU via DDP (`torchrun`). The final model checkpoint is saved under `models/nrows_None__nsrc_None/timestamp_<ts>/bge-base-en-v1.5/final_model/`.

`src/utils/utils.py` contains the binarization layers:
- `BinarizationLayer`: Straight-Through Estimator — `sign(x)` forward, identity backward.
- `AnnealedTanhBinarizationLayer`: training uses `tanh(β·x)` where `β = sqrt(γ·step + 1)` (β grows from 1 → ~28 over 8k steps); inference uses `sign(x)`. `BetaAnnealCallback` in `train.py` calls `.anneal_step()` after every optimizer step and logs β to W&B.

### Evaluation system (`eval_v2/`)

**Embedding cache** (`eval_v2/core/cache.py`): Raw float32 embeddings are encoded once and stored at `eval_v2/outputs/cache/<sha256(model_path)[:16]>/nanobeir/<Subset>/{corpus,queries}.{npz,json}`. Multiple `ModelSpec` entries that share the same checkpoint path reuse the same cache. The `.json` sidecar stores metadata (model path, n, dim). A cache miss triggers multi-GPU encoding.

**Transform pipeline** (`eval_v2/wrappers/transforms.py`): Applied at load time, after the cache, in this order:
1. Truncate dimensions (`truncate_dim`)
2. Binarize → uint8 packed (`similarity == "hamming"`)
3. INT8/INT4 quantize (`quant_bits`)
4. PQ reconstruct — fit on corpus, asymmetric (`pq_M`)
5. TurboQuant dequantize — fit on corpus (`tq_bits`)

**Model registry** (`eval_v2/config/models.py`): `MODELS: dict[str, ModelSpec]` is the single source of truth for every experiment variant. `ModelSpec` fields control which transforms are applied at eval time — there is no separate config file. Models that are post-hoc transforms of the same checkpoint (INT8, PQ, TQ, MRL truncations) share one cache entry and differ only in their `ModelSpec`.

**Evaluator** (`eval_v2/core/evaluator.py`): Scores corpus/query embedding pairs using cosine or Hamming similarity, computes NDCG@k, MRR@k, Recall@k for k∈{1,3,5,10}, runs paired significance tests across model pairs, and emits `SubsetResult` objects.

### Adding a new experiment

1. **Train**: add experiment branch in `src/train.py` (follow e5 pattern), run with new `--expirement_number`.
2. **Register**: add one or more `ModelSpec` entries to `MODELS` in `eval_v2/config/models.py` pointing to the new checkpoint path. If the experiment is a post-hoc transform of an existing checkpoint, no new training is needed — just a new spec.
3. **Evaluate**: `python -m eval_v2.run --models your_new_key`.

## Thesis Report Writing Guidelines

- Do not use em dashes (—) or double/triple hyphens (---, --) in prose. Use commas, semicolons, or restructure the sentence instead.
- When stating a fact or claim, cite the source. Use the existing citation style of the document (e.g. `\cite{key}` in LaTeX).
- On the first occurrence of any term, write the full form followed by the abbreviation in parentheses — e.g., "Information Retrieval (IR)". After that first instance, use the abbreviation alone.
- Keep sentences short and easy to read. If a sentence is getting long or complex, break it into two or more shorter sentences.

### Multi-bit annealed tanh (planned e6)

The `experiments/notes.md` outlines the research gap: combine a differentiable staircase quantizer (sum of `B-1` shifted tanh/sigmoid functions) with annealed temperature, trained end-to-end for retrieval. Key references: Quantization Networks (CVPR 2019), QAMA (CIKM 2025). The `AnnealedTanhBinarizationLayer` in `src/utils/utils.py` is the direct extension point; `ModelSpec` would need a new `atanh_bits` field.
