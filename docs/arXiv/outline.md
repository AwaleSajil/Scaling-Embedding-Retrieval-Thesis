# arXiv Paper Outline

Working title, abstract, and section-by-section plan for turning the thesis into a
conference/arXiv paper. The organizing principle: **lead with the effective-dimension
finding, not with "controlled comparison."** The comparison is the scaffolding; the
mechanism is the contribution people cite.

---

## 1. Framing decision (read first)

The thesis has two stories. Pick the lead deliberately.

- **Story A (forgettable lead):** "We ran a controlled comparison of post-hoc vs
  training-time vs stacked compression. No method wins everywhere; pick by backbone and
  budget." True, useful, but reads as a benchmark study.
- **Story B (citable lead):** "Binarization-aware training can *beat* full-precision
  retrieval on encoders without retrieval pre-training, because binarization raises the
  embedding space's effective dimension. We explain the mechanism and map when it holds."

**Recommendation: lead with Story B; use Story A as the supporting structure.** The
Pareto/no-universal-winner result becomes the practical payoff in the second half, not the
headline.

### Title candidates
1. *When Compression Helps: Binarization-Aware Training Increases Effective Dimension in Dense Retrieval*
2. *No Universal Winner: A Controlled Study of Embedding Compression Across Retrieval Backbones*
3. *Binarization as Regularization: Effective Dimension Explains When Compression Beats Full Precision*

(1) or (3) if leading with the mechanism; (2) if the venue wants a benchmark framing.

---

## 2. Abstract (draft, ~150 words)

> Dense retrieval stores large float32 embedding indices in RAM, which dominates serving
> cost at scale. We present a controlled comparison of embedding compression for retrieval:
> three encoders with different retrieval priors (BGE, RoBERTa, MPNet) are fine-tuned under
> identical conditions and compressed with post-hoc methods (INT8, binary, Product
> Quantization, TurboQuant), training-time binarization (Straight-Through Estimator,
> annealed tanh), Matryoshka Representation Learning, and stacked combinations. We find no
> method dominates across backbones: post-hoc PQ leads the Pareto frontier for
> retrieval-pretrained BGE, while training-time binarization leads for RoBERTa and MPNet.
> Most strikingly, binarization-aware training *exceeds* the full-precision baseline on
> RoBERTa. We trace this to effective dimension: binarization spreads embedding variance
> across more axes (participation ratio 11%->30%), improving ranking without changing
> recall. The gain is structural, not under-training. We release a Pareto cheat-sheet for
> selecting a method given a backbone and memory budget.

Tighten to the venue's word limit. Keep the "exceeds full precision" + "effective
dimension" sentences no matter what; they are the hook.

---

## 3. Section-by-section plan

Each section notes: **what it argues**, **source material in the thesis**, and **what's
missing / to add**.

### 1. Introduction
- **Argues:** memory is the bottleneck for dense retrieval at scale; the compression
  literature is fragmented across backbones/benchmarks so practitioners can't choose; we
  do a controlled study and find a surprising mechanism.
- **Source:** `Chapters/ch1_introduction.tex` (motivation, the 2.9 TB / $3.80 per GB-month
  cost table).
- **Add:** state the 3-4 contributions as a bullet list, with the effective-dimension
  finding first. Preview the headline number (BAT 0.590 > float32 0.530 on RoBERTa).
- **RQs (keep the thesis's four):** post-hoc vs training-time; STE vs annealed tanh;
  does stacking help; do findings generalize across backbones.

### 2. Related Work
- **Argues:** position against the four closest works and name the gap precisely.
- **Source:** `ch2_background.tex`, `related_works_review.md`, slides 27-30 (the gap matrix
  is good — reuse it as a figure/table).
- **Key contrasts:**
  - CoRECT (2026): closest *evaluation* study, but post-hoc only, no training-time.
  - QAMA (CIKM 2025): closest *method*, MRL + multibit QAT, but penalty loss not soft
    staircase, no tanh.
  - BPR (Yamada, ACL 2021): annealed tanh vs PQ, but never vs STE.
  - BEBR (Gan, KDD 2023): recurrent residual binarization, not end-to-end.
  - INDUS-SDE (Pantha, KDD 2026, *our prior work*): BAT-STE > post-hoc binary, but single
    model/domain, STE only.
- **Gap sentence:** no prior work compares post-hoc *and* training-time *and* stacked
  compression across backbones with differing retrieval priors, nor explains the
  backbone-dependent behavior.

### 3. Method / Experimental Design
- **Argues:** the comparison is fair (same backbone init, same data, same benchmark, same
  hyperparameters) and broad (226 configs).
- **Source:** `ch4_methodology.tex`, slides 32-42.
- **Cover:** bi-encoder + MNRL; the 6 training variants; the 4 post-hoc transforms; MRL;
  the eval pipeline (embedding cache + transform order); Wilcoxon significance protocol.
- **Add:** keep this tight for a paper. The cache/transform-pipeline detail can be
  compressed to a paragraph + appendix.

### 4. Results
Order the subsections so the build-up pays off at 4.4.

- **4.1 Float32 baselines** — establish ceilings; note BGE fine-tuning is a no-op
  (`p=0.49`), RoBERTa/MPNet gain hugely. (`tab:float32_baselines`)
- **4.2 Post-hoc compression** — INT8 near-lossless; binary loses 3-5 pp; PQ near-lossless
  to 96x; TurboQuant 4-bit lossless. Cross-method at 32x: PQ > TQ-1bit > binary, stable
  across backbones. (`tab:int8`, `tab:binary`, `tab:tq`, `tab:cross_method_32x`)
- **4.3 Training-time binarization** — BAT beats post-hoc binary everywhere; gains largest
  on non-retrieval backbones. (`tab:ste_atanh`)
- **4.4 THE HEADLINE: why BAT beats float32 on RoBERTa** — give this the most space.
  - 4.4.1 The gain is ranking, not coverage (MRR up, Recall@10 flat, with p-values).
  - 4.4.2 The mechanism is effective dimension (`tab:effdim`, participation ratio
    11%->30%; the cumulative-variance figure).
  - 4.4.3 Why it's RoBERTa-specific (BGE already high d_eff, nothing to correct).
  - 4.4.4 It's structural, not under-training (training-trajectory figure).
- **4.5 MRL and stacked combinations** — MRL alone collapses past 128-d; MRL+PQ dominates
  >=96x; MRL+atanh > MRL+STE across all backbones. (`tab:mrl`, stacked figure)
- **4.6 Pareto frontiers + cross-backbone** — the "no universal winner" synthesis; the
  cheat-sheet. (`fig:pareto`, `fig:cross_backbone`)

### 5. Discussion
- Binary constraints as regularization (tie back to d_eff).
- Practical guidance: recommended method by bit budget and backbone (the cheat-sheet).
- Threats to validity: NanoBEIR scale, single epoch, no latency numbers — state these
  honestly here so reviewers see you know.

### 6. Conclusion
- Source: `ch6_conclusion.tex`. Compress to ~1 paragraph + future work pointer.

---

## 4. Figures and tables to carry over (already exist)

- `fig_pareto.pdf` — the money figure. Pareto frontiers, 3 backbones.
- `fig_roberta_effdim.pdf` + `tab:effdim` — the mechanism. Promote to main text.
- `fig_training_trajectory.pdf` — kills the under-training objection.
- `fig_stacked_combinations.pdf` — MRL+PQ dominance.
- `fig_crossmethod_32x.pdf`, `fig_cross_backbone.pdf` — supporting.
- Research-gap matrix (slide 30) — redraw as a clean table.

Likely move to appendix: per-method quality-vs-ratio panels (PQ, TQ), gamma ablation,
significance-test details.

---

## 5. Gaps to close before submission (priority order)

1. **[HIGH] Scale beyond NanoBEIR.** Re-run the existing pipeline on a few *full-size*
   BEIR datasets (e.g. NQ, FiQA, SciFact, Touche2020). The eval system already caches
   embeddings, so this is mostly compute. Goal: show the backbone-dependence and the d_eff
   finding survive at realistic corpus sizes. This is the difference between workshop and
   main track, and it directly answers the CoRECT-style "does it hold at scale?" objection.
2. **[HIGH] Latency + on-disk index size.** Report actual query latency (cosine vs Hamming)
   and measured index size. The motivation is cost; close the loop with measured cost.
3. **[MED] Strengthen the d_eff causal claim.** The noise-injection experiment proposed in
   `ch6` future work (add noise to BGE, watch BAT's relative benefit rise) would turn a
   correlation into evidence of a mechanism. High payoff if it works.
4. **[LOW] e6 (multi-bit annealed tanh).** Out of scope for this paper; do not position as
   a methods contribution. Save for a follow-up.

---

## 6. Venue targets

| Effort | Venue |
|--------|-------|
| As-is (NanoBEIR) | ReNeuIR @ SIGIR, ECIR short, EMNLP/ACL Findings |
| + full BEIR + latency | SIGIR / ECIR / CIKM main track |

The single highest-leverage experiment is gap #1. Everything else is polish.

---

## 7. Open decisions for the author

- Lead with mechanism (Story B) or comparison (Story A)? (Recommend B.)
- Include MPNet as a third backbone or cut to BGE+RoBERTa for a tighter narrative? (Three
  strengthens the cross-backbone claim; keep all three.)
- Target an IR venue (SIGIR/ECIR/CIKM) or an NLP venue (EMNLP/ACL Findings)? IR venues
  reward the Pareto/systems framing; NLP venues reward the mechanism. The d_eff story
  plays well in both.
