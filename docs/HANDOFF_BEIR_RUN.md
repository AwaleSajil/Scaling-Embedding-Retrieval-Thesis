# Handover: full BEIR evaluation

Written 2026-07-29 by the original owner (`sawale`) before losing machine access.
Read this end to end before touching anything.

## What is running

A full BEIR evaluation of 224 embedding-compression configurations across all
13 BEIR subsets — 2,912 `(model, subset)` pairs, ~954M embeddings to encode.

It runs as a SLURM array on the `matrix` cluster: one task per subset, smallest
first, serialised with `%1`. A dependent finalize job computes significance
tests and builds the dashboard once every task succeeds.

```
array job     3924965   (13 tasks, 0-12)
finalize job  3924967   (afterok:3924965)
```

State at handover: **9 of 13 subsets complete, 2016/2912 pairs (69%)**.
Remaining: hotpotqa (running), fever, climate-fever, msmarco. Roughly 7 days.

## Where everything lives

| What | Path | Access |
|---|---|---|
| Code | `git@github.com:AwaleSajil/Scaling-Embedding-Retrieval-Thesis.git`, branch `develop` | public-ish, clone it |
| Embedding cache | `/nas/rgroup/dsig/llm-team/sajil-thesis/eval_cache` (504 GB and growing, ~2.9 TB final) | group `dsig`, rwx |
| Results | `/nas/rgroup/dsig/llm-team/sajil-thesis/results/beir` | group `dsig`, rwx |
| HF datasets | `/nas/rgroup/dsig/llm-team/sajil-thesis/hf_cache` | world-readable |
| SLURM logs | `/rhome/sawale/thesis/eval_v2/slurm_logs/` | **owner only** |
| Checkpoints | `/rhome/sawale/thesis/models/...` | **owner only** |
| conda env | `/rhome/sawale/miniconda3/envs/thesis_eval` | **owner only** |
| Secrets | `/rhome/sawale/thesis/.env` (HF + W&B tokens) | **owner only** |
| Monitoring | https://wandb.ai/impact-ibm-collaboration/beir-eval | W&B team project |

Commit at handover: `e8e6fa9`.

## THINGS THE NEW OWNER MUST DO FIRST

### 1. Get at the checkpoints

`/rhome/sawale` is mode 0750, so nothing under it is readable by anyone else.
The 29 model checkpoints live there and are needed **only for encoding**. Once a
subset's embeddings are cached, scoring and finalize never load a model.

If encoding is still unfinished, the outgoing owner must grant traversal before
leaving:

```bash
chmod o+x  /rhome/sawale /rhome/sawale/thesis
chmod -R o+rX /rhome/sawale/thesis/models
chmod o+rx /rhome/sawale/thesis/eval_v2/slurm_logs
```

`o+x` on the home directory permits traversal but not listing, so it does not
expose unrelated files.

**Do NOT copy the checkpoints elsewhere and repoint `models.py`.** The embedding
cache is keyed on `sha256(model_path)`, so changing a path invalidates the
matching cache entries and forces a re-encode of everything — days of GPU time.
There is a `models.zip` (88 GB, dated June 2026) on the dsig volume, but it
predates some checkpoints and its paths would have the same problem.

### 2. Build your own environment

`thesis_eval` is in the old owner's home. Recreate it under your own account.
**`eval_v2/SETUP.md` is wrong for this cluster** — follow this instead:

```bash
conda create --name thesis_eval --clone thesis -y     # or build from scratch
conda activate thesis_eval
pip install --only-binary=:all: "faiss-cpu==1.9.0.post1" "turboquant==0.2.0" "pyarrow==20.0.0"
```

Why these exact pins, all learned the hard way:

- `conda install -c conda-forge faiss-cpu` **cannot work**: the base env is from
  the `defaults` channel (`_libgcc_mutex 0.1 main`) and conda-forge's faiss
  requires `_libgcc_mutex 0.1 conda_forge`. Unsatisfiable.
- `faiss-cpu` newer than 1.9.0.post1 has no cp312 manylinux2014 wheel, so pip
  falls back to an sdist that tries to build numpy from source and fails.
  matrix is CentOS 7.9 / glibc 2.17.
- `pyarrow` 19.0.0 cannot read HuggingFace's current BEIR parquet files
  (`Repetition level histogram size mismatch`). 20.0.0 is the newest with a
  cp312 manylinux2014 wheel — there is no upgrade path beyond it on CentOS 7.

### 3. Your own tokens

Create `.env` at the repo root with your own credentials:

```
HUGGINGFACE_TOKEN=...
WANDB_API_KEY=...
```

Nothing in the run depends on the previous owner's accounts except the W&B
project name, and that lives under a team entity you can point elsewhere with
`--wandb_project`.

## Machines

- **matrix** (`ssh <you>@matrix`) — CentOS 7.9, glibc 2.17. Has SLURM. The only
  GPU node is `matrix101`: 3x A100 80GB, 64 CPUs, 382 GB. Shared with others.
- **dsig1** — RHEL 9. No SLURM. Mounts the same NAS. Use it for monitoring and
  analysis so you are not holding a session on the cluster.

The repo's uv `.venv` works on dsig1 but **cannot run on matrix at all** —
torch 2.8 needs glibc >= 2.28. matrix must use conda.

dsig1's GPUs are a V100 (sm_70) plus two P100s (sm_60); torch 2.8 has no sm_60
kernels, so the P100s fail with "no kernel image is available". Scripts here
filter GPUs by torch's compiled arch list.

## Monitoring

```bash
# from dsig1
cd <repo> && .venv/bin/python -m eval_v2.watch_progress --watch 300
```

Reads durable state (cache `.npz` files, `aggregate.json`) rather than parsing
logs, and reports per subset. Totals are 754 encodes and 2,912 pairs.

```bash
# on matrix
squeue -u <you>
sacct -j <arrayjob> --format=JobID,State,Elapsed,MaxRSS,ExitCode
tail -f /path/to/slurm_logs/<arrayjob>_<task>_beir.out
```

W&B shows one run per subset, named `beir-<subset>-<arrayjob>_<task>`, so a run
maps directly onto its SLURM log. A crashed job shows as `crashed` there.

## If something fails

**Everything is resumable.** `store.already_evaluated()` keys on
`(model, subset)` and the embedding cache persists, so simply resubmitting skips
all completed work. A failed task costs only its own remainder.

```bash
cd <repo>/eval_v2
unset EXT_DISK BEIR_MODELS          # see gotcha 1
AID=$(sbatch --parsable eval_beir_array.slurm)
sbatch --dependency=afterok:$AID eval_beir_finalize.slurm
```

To rerun a single subset, `--array=N` with the index from the `SUBSETS` list in
`eval_beir_array.slurm` (0=nfcorpus … 12=msmarco).

If a subset is genuinely stuck and you want the rest published, submit finalize
on its own — it computes significance and the dashboard from whatever is on
disk, and prints a coverage report first so you can see what is missing:

```bash
sbatch eval_beir_finalize.slurm
```

## Gotchas that have already bitten this run

1. **`sbatch` defaults to `--export=ALL`.** A stale `EXT_DISK` or `BEIR_MODELS`
   exported in your shell (from smoke testing) silently redirects the real run
   to a throwaway tree and/or four specs. The job looks healthy. Always `unset`
   both first; the script now echoes `EXT_DISK` and warns if it is not the
   default.

2. **A crashing eval used to report success.** The array script had no exit-code
   propagation, so a Python crash still exited 0, SLURM said `COMPLETED`, and
   the finalize job's `afterok` was satisfied — significance would have been
   computed over incomplete results. Fixed in `e8e6fa9`; do not remove that
   guard.

3. **TurboQuant allocates ~2^bits floats per element.** At 8 bits that is 256x
   the corpus. It asked for 300 GB on webis-touche2020 and killed the job,
   losing 44 specs. `_apply_tq` now chunks at 4096 rows. Chunking is exact only
   if no chunk is tiny — a 57-row tail changes results — so short tails are
   merged into the previous chunk. Verified bit-identical.

4. **`--finalize` is not the same as re-running.** A plain rerun reloads every
   dataset and calls `get_or_compute` for all 29 checkpoints (~2.9 TB of NAS
   reads) before skipping already-done pairs.

5. **Per-subset workers each wrote a wrong `mean`.** `save_mean_metrics()` takes
   the subsets of the current job, so each worker overwrote `mean` with its own
   subset alone. `--finalize` recomputes means across everything present. If you
   read `aggregate.json` mid-run, the `mean` key is meaningless until finalize
   has run.

6. **Results are not bit-reproducible across machines.** dsig1 and matrix differ
   slightly on PQ and 1-bit TurboQuant specs (different faiss build and BLAS,
   plus heavy score ties). Differences are ~1e-3 on ndcg@10. Do not treat a
   small mismatch as a bug — compare like with like.

## Known outstanding issues

- **fever and climate-fever share a corpus** (identical modulo 25 documents) but
  are encoded twice because the cache keys on `(model, subset)`. Deduplicating
  saves ~40 hours of GPU time with no effect on results. Not done.
- **Every spec re-reads its checkpoint's cache**, so the run does ~22.6 TB of
  NAS reads instead of ~2.9 TB. Grouping the evaluation loop by checkpoint would
  cut ~11 hours. Not done.
- **`significance.json` will be several GB** and `compute_significance()` loads
  every model's per-query shard into one dict — the pipeline's memory peak,
  and it happens last. The finalize job asks for 200 GB for this reason.
- **Significance is single-threaded** — a plain nested Python loop over ~5.6M
  paired tests, roughly 3-5 hours. More CPUs will not help unless it is
  parallelised first.
- **Scoring never uses the GPU.** `run_ir_eval` is entirely CPU tensors, so the
  A100s idle through the scoring phase. Moving it to GPU is the largest
  remaining speedup and has not been attempted.

## Publishing the results

The dashboard is a self-contained `index.html` plus shard directories:

```bash
python -m eval_v2.hf_deploy_dashboard \
    --results_dir /nas/rgroup/dsig/llm-team/sajil-thesis/results/beir \
    --repo_id <you>/Embedding-Compression-Eval-BEIR --private
```

Only `index.html`, `README.md`, `plotly.min.js`, `per_query_shards/*` and
`significance_shards/*` are uploaded — the embedding cache is an intermediate
and never leaves the cluster. Note `plotly.min.js` is not produced by the eval;
the deploy script copies it from an existing results directory.

Expect the BEIR dashboard to be far larger than the NanoBEIR one (1.8 GB): the
msmarco per-query shard alone projects to ~1.5 GB, which a browser would try to
fetch on a single click. Subsampling per-query data may be necessary.
