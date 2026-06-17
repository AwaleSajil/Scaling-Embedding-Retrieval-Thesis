# TODO: update report with "why RoBERTa-BAT beats fp32" analysis

Status: WRITTEN INTO REPORT (2026-06-11). Core analysis added to ch5
(subsec:ste_atanh), replacing the old "left as future work" punt. Build verified (95 pp).
Created: 2026-06-08

## Where it went
- ch5_expirements_results.tex, end of the STE/Annealed-Tanh subsection.
  Four paragraphs: ranking-not-coverage, effective-dimension mechanism,
  why-RoBERTa-specific, structural-not-under-training.
- New table `tab:effdim` (eff-dim + MRR/Recall, 5 models).
- New figure `fig:roberta_effdim` -> Figures/fig_roberta_effdim.pdf
  (cumulative explained variance, sourced from presentation slide 51 / image38).
- Bib added to ref.bib: ethayarajh2019contextual, litwinkumar2017optimal, gao2017theory.

## Action items
- [x] Effective-dimension table added as LaTeX (tab:effdim).
- [x] Variance-spectrum figure added (fig_roberta_effdim.pdf, cumulative-variance panel).
- [x] Prose argument written (chain of evidence below).
- [x] Participation-ratio method + citations stated.
- [~] Significance reported as inline p-values (STE p=0.005, atanh p<0.001; recall n.s.).
      NOT yet rendered as n.s./*/**/*** markers on the bar-chart figures.
- [ ] MRR-vs-Recall per-subset delta figure (appendix) still not added; key
      per-subset numbers (Touche2020/FEVER/DBPedia) are in prose instead.

## Variance-spectrum figure (DONE - generated)
- File: `analysis/figures/variance_spectrum_effdim.png`
- Code: cell "Plot 1b" in `analysis/7_anisotropy.ipynb` (re-run to regenerate).
- Two panels: (A) PCA variance spectrum (log-y, first 250 PCs) - steeper decay = more collapsed;
  (B) cumulative explained variance with dotted effective-dimension markers per model.
- Same 5 models as the table; d_eff annotated in the legend. Use panel B as the main "variance plot"
  for intuition, then hand off to the eff-dim table for the cross-backbone comparison.

## The argument (chain of evidence)

**Claim:** On RoBERTa, binarization-aware training (STE and atanh) beats uncompressed fp32 on retrieval at 32x compression. Not seen on BGE.

1. **Significant, not noise.** Paired Wilcoxon over 649 queries: BAT-STE vs fp32 p=0.005; BAT-atanh vs fp32 p<0.001 (MRR@10).

2. **Ranking gain, not coverage.** MRR@10 improves significantly; Recall@10 is unchanged (p=0.57 STE, p=0.27 atanh; effect ~0). Same docs retrieved, ordered better = re-ranking effect.

3. **Broad & consistent.** MRR improves on 11/13 NanoBEIR subsets; recall deltas scatter around zero. On biggest-MRR-gain subsets (Touche2020 +0.185, FEVER +0.105, DBPedia +0.096) recall barely moves.

4. **Mechanism = effective dimension (participation ratio).** RoBERTa fp32 space is collapsed (d_eff = 85/768 = 11%, most anisotropic backbone). Binarization spreads variance across axes, raising binary d_eff. Binary d_eff tracks MRR monotonically.

5. **Why only RoBERTa.** BGE already high d_eff (17.6%) + retrieval-trained -> nothing to fix, binary even costs it. MPNet intermediate (13.3%, near-lossless). Benefit largest where float space most collapsed.

6. **atanh > STE.** atanh = most isotropic binary code (highest d_eff) = best retrieval.

7. **Structural, not under-training.** Gap analysis: BAT-vs-fp32 advantage does NOT close as training continues.

## Effective-dimension table (1-epoch final models, pooled NanoBEIR corpus)

FT rows = float d_eff + cosine retrieval. BAT rows = binary d_eff + Hamming retrieval.

| Model | Effective dim | % of 768 | MRR@10 | Recall@10 |
|---|---|---|---|---|
| RoBERTa-FT | 85 | 11.0% | 0.530 | 0.553 |
| BGE-FT | 135 | 17.6% | 0.685 | 0.656 |
| MPNet-FT | 102 | 13.3% | 0.582 | 0.583 |
| RoBERTa-BAT-STE | 197 | 25.7% | 0.565 | 0.552 |
| RoBERTa-BAT-atanh | 231 | 30.1% | 0.590 | 0.560 |

RoBERTa binary-code progression: post-hoc binary 25.0% / MRR 0.501 -> STE 25.7% / 0.565 -> atanh 30.1% / 0.590.

## Participation ratio (method, for the report)
d_eff = (sum lambda_i)^2 / (sum lambda_i^2), where lambda_i are the 768 eigenvalues of the
centered embedding covariance (= variance per PCA axis). Equivalently d_eff = 1 / sum(p_i^2)
with p_i = lambda_i / sum(lambda). Counts effective number of variance-carrying dimensions:
=1 if all variance in one axis, =768 if spread evenly. Code: `pca_stats` in
`analysis/7_anisotropy.ipynb`; over-steps version `analysis/scripts/anisotropy_over_steps.py`.

## Data sources
- Significance: `eval_v2/outputs/results/nanobeir/significance.json` (keys: modelA|||modelB|||subset|||metric; use subset "mean").
- Aggregate scores: `eval_v2/outputs/results/nanobeir/aggregate.json`.
- Effective dim recomputed from eval caches under `eval_v2/outputs/cache/<hash>/nanobeir/*/corpus.npz`.
- Model keys: RoBERTa-FT=`finetuned-roberta-base`, STE=`finetuned-roberta-base-bat` (e2), atanh=`finetuned-roberta-base-atanh-gamma0.1` (e5).
