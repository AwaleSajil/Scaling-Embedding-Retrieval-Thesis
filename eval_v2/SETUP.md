# eval_v2 environment setup

The eval system needs three libraries **beyond** the training `thesis` conda env
(which already provides `torch`, `sentence-transformers`, `datasets`,
`transformers`, `numpy`, `python-dotenv`, etc.).

| Library | Why eval needs it | In training env? | Install via |
|---------|-------------------|------------------|-------------|
| **faiss-cpu** | 90 Product-Quantization (PQ) variants (`core/evaluator.py`) | No (missing everywhere) | conda-forge |
| **turboquant==0.2.0** | 42 TurboQuant variants (`core/evaluator.py`) | No (eval-only) | pip |
| **scipy** | Significance tests (`core/significance.py`) | Probably (via sklearn/transformers) | conda-forge if missing |

`faiss` is the critical gap: without it, all 90 PQ specs crash *after* encoding,
wasting GPU hours. `faiss-cpu` is sufficient (PQ runs on CPU); `faiss-gpu` is not
needed.

## Old server notes

1. **Install faiss from conda-forge, not pip.** The pip `faiss-cpu` wheels assume
   a recent glibc/libstdc++ and fail with `GLIBCXX_... not found` on older
   systems. The conda-forge build bundles its own libstdc++ and avoids this.
2. **Clone the training env instead of modifying it.** Adding eval deps can make
   conda re-solve and silently bump `numpy`/`torch`, which could break training.
   Clone first, then add eval deps to the clone.

## Install

```bash
# 1. Clone the working training env so training stays untouched
conda create --name thesis_eval --clone thesis
conda activate thesis_eval

# 2. faiss — conda-forge build (reliable on old glibc)
conda install -c conda-forge faiss-cpu

# 3. scipy — only if the clone doesn't already have it
conda install -c conda-forge scipy

# 4. turboquant — pip-only package, exact version
pip install turboquant==0.2.0
```

## Verify

```bash
conda activate thesis_eval
python -c "import faiss, scipy, turboquant, sentence_transformers, datasets; \
print('faiss', faiss.__version__, '| scipy', scipy.__version__, '| all eval deps OK')"
```

If you use the `thesis_eval` clone, update the SLURM script activation line in
`eval_v2/eval_beir_full.sh` from `conda activate thesis` to
`conda activate thesis_eval`.
