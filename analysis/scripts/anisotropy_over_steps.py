"""
Encode the pooled NanoBEIR corpus with each per-step RoBERTa checkpoint
(baseline / BAT-STE / atanh) using ONLY [Transformer, Pooling] so we capture
the genuine PRE-SIGN float embedding, then save the PCA eigen-spectrum.

The sign() spectrum is derived later for free as np.sign(X).

Run:  CUDA_VISIBLE_DEVICES=0 python analysis/scripts/anisotropy_over_steps.py
Outputs: analysis/figures/anisotropy_steps/spectra.npz  (+ corpus cache)
"""
import os, sys, pickle, hashlib
from pathlib import Path
import numpy as np

ROOT = Path("/nas/rhome/sawale/thesis")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))          # custom binarization modules
sys.path.insert(0, str(ROOT / "eval_v2"))

OUT = ROOT / "analysis/figures/anisotropy_steps"
OUT.mkdir(parents=True, exist_ok=True)
CORPUS_PKL = OUT / "pooled_corpus_texts.pkl"
MAX_ROWS = 20_000          # subsample pooled corpus for the spectrum
SEED = 42

RUNS = {
    "baseline": "models/nrows_None__nsrc_None/timestamp_20260606_11-12-30/roberta-base",
    "BAT-STE":  "models/nrows_None__nsrc_None/timestamp_20260606_11-15-44/roberta-base",
    "atanh":    "models/nrows_None__nsrc_None/timestamp_20260606_11-17-10/roberta-base",
}
STEPS = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 16378]


def get_corpus_texts():
    if CORPUS_PKL.exists():
        return pickle.load(open(CORPUS_PKL, "rb"))
    from eval_v2.run import load_nanobeir_subset
    from eval_v2.config.datasets import NANOBEIR_SUBSET_TO_HF
    tok = os.environ.get("HUGGINGFACE_TOKEN")
    texts = []
    for sub in NANOBEIR_SUBSET_TO_HF:
        _, ctext, *_ = load_nanobeir_subset(sub, hf_token=tok)
        texts.extend(ctext)
        print(f"  {sub}: +{len(ctext)} (total {len(texts)})", flush=True)
    pickle.dump(texts, open(CORPUS_PKL, "wb"))
    return texts


def subsample(texts):
    rng = np.random.default_rng(SEED)
    if len(texts) > MAX_ROWS:
        idx = rng.choice(len(texts), MAX_ROWS, replace=False)
        return [texts[i] for i in idx]
    return texts


def encode_presign(ckpt_path, texts):
    """Load checkpoint, drop any binarization head, encode pre-sign float."""
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(ckpt_path, device="cuda")
    # keep only Transformer + Pooling
    keep = [m for m in st if type(m).__name__ in ("Transformer", "Pooling")]
    base = SentenceTransformer(modules=keep, device="cuda")
    X = base.encode(texts, batch_size=64, show_progress_bar=False,
                    convert_to_numpy=True, normalize_embeddings=False)
    del st, base
    import torch, gc
    gc.collect(); torch.cuda.empty_cache()
    return X.astype(np.float32)


def spectrum(X):
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = (s ** 2) / (len(X) - 1)
    return lam.astype(np.float32)


def pr_from_lam(lam):
    tot = lam.sum()
    return float((tot ** 2) / np.sum(lam ** 2))


def main():
    print("Pooling corpus ...", flush=True)
    texts = subsample(get_corpus_texts())
    print(f"corpus rows used: {len(texts)}", flush=True)

    results = {}   # f"{variant}@{step}@{float|sign}" -> lam
    summary = []
    for variant, run in RUNS.items():
        for step in STEPS:
            ck = ROOT / run / "checkpoints" / f"checkpoint-{step}"
            if not ck.is_dir():
                print(f"MISSING {variant}@{step}", flush=True); continue
            X = encode_presign(str(ck), texts)
            lam_f = spectrum(X)
            lam_s = spectrum(np.sign(X))
            results[f"{variant}@{step}@float"] = lam_f
            results[f"{variant}@{step}@sign"] = lam_s
            prf, prs = pr_from_lam(lam_f), pr_from_lam(lam_s)
            summary.append((variant, step, prf, prf/X.shape[1], prs))
            print(f"{variant:9s} step {step:5d}  eff-dim float={prf:6.1f} "
                  f"({100*prf/X.shape[1]:4.1f}%)  sign={prs:6.1f}", flush=True)

    np.savez_compressed(OUT / "spectra.npz",
                        **results,
                        meta=np.array([f"{v}|{s}|{pf}|{fr}|{ps}"
                                       for v, s, pf, fr, ps in summary]))
    print(f"\nsaved -> {OUT/'spectra.npz'}", flush=True)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    main()
