# Introduction

- Motivation paragraph (broad context and why: the hook)
	- Semantic search  / dense retrival has been on the rise especially with RAG system on the rise
	- but as the scale of documents increases (to millions, billion ,... ), the embedding size also increase. 
	- semantic search system need these embedding to be on memory  (why we need it to be in memoery find a academic source maybe)
	- Having a system with large memeory can be expensive
	- $3.8 per GB/month on AWS x2gd instance (memeory optimized aws instance)
	- use the https://huggingface.co/blog/embedding-quantization or other academic source
	- one motivation to have better compression is cost another is latency
	- you can give example when we use binary embeddings we don't have to use cosine distance, we can use hamming distance which could be done in one cpu ... (x times faster) (may be find some academic source for this claim)
	- although we don't do much experiments on latency in this thesis

- Core Tension Paragraph
	- There are bunch of techniques (post ad hoc / online (after we have the embeddings) or training time based model learns to compress represenation during training) ) to compress the embeddings (shoudl i give examples here ? or in literature review ?)
	- But the retrival performance degrades by some factor 
- Research question
    - Does vector quantization aware trained embedding model outperform post-hoc vector quantization techniques ? (How do training-time methods compare against post-hoc baselines?)
	- Which method is the most effective at compression and maintain reterival performance
	- How do post-hoc compression methods compare at various bit budgets
	- Does combination of compression method (Mostly with MRL) yield better compression to retrival performance than 
	- Does Annealed Binarization outperform the Straight-Through Estimator baseline (BAT)?
- Contribution
	- List the 5–6 novel contributions
- Thesis Organisation
	- TBD


Literature Review





---
# Interesting things found / to concider

## Why BGE-base dropped slightly on NanoBEIR after fine-tuning:
BGE is already trained on 100M+ retrieval pairs — it's at a retrieval-specialized optimum.
Fine-tuning on your 8-dataset mix introduces distribution shift without enough signal to overcome
it, causing the model to partially forget its broad generalization. This is well-documented: dense
models fine-tuned on one domain systematically underperform BM25 on out-of-domain subsets, with
in-domain performance not correlating with generalization capability
[(Thakur et al., 2021)](https://arxiv.org/abs/2104.08663). This trade-off is further confirmed by
[(Yu et al., 2022)](https://arxiv.org/pdf/2210.15212), who show that full fine-tuning achieves
higher in-domain but lower out-of-domain performance, and that hard negative mining boosts
in-domain performance at the cost of OOD generalization — also echoed by
[(Zhan et al., EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.481.pdf). The
non-significance on Wilcoxon is expected — the drop is real but marginal because BGE already
covers most of NanoBEIR's distribution. [(Shen et al., 2024)](https://arxiv.org/html/2410.12890)
quantify this effect and note drops are often within noise range when the base model is already
well-generalized.

Importantly, this drop does not indicate poor training data. The drop would likely occur even
with perfect training data, because BGE's pre-training already covers a superset of retrieval
knowledge. Any fine-tuning on a smaller distribution shifts the model away from that broad
optimum, necessarily de-weighting some previously learned patterns. Evidence for this
interpretation is that the drop is non-significant, and limited to out-of-domain evaluation —
what would indicate genuinely bad training data would be a large, significant drop or failure
even on in-domain tasks [(Yu et al., 2022)](https://arxiv.org/pdf/2210.15212).

**RoBERTa improved because** it started from MLM (a generation task) — fine-tuning with MNRL is
a large task-alignment jump for it, whereas for BGE it is redundant noise.


## Why RoBERTa shows a clear difference between binarization strategies

Both BAT and post-hoc binarization start from the **base pretrained model**
and fine-tune with MNRL. The difference is only whether the binarization
constraint is active *during* fine-tuning or applied *after*.

**For BGE:** BGE-base is already retrieval-specialized — its representations
are already directionally organized (MNRL is a directional objective;
magnitude is normalized away). When `sign()` is added during BAT, the loss
spike is **small** because the sign pattern already captures most retrieval
signal even from the base model. The gradient does not push the model far
from where standard fine-tuning would land.
Result: BAT ≈ post-hoc binarization.

**For RoBERTa:** RoBERTa-base is MLM-specialized — its representations
encode information in both magnitude and direction, with no directional bias
toward retrieval. Adding `sign()` during BAT causes a **large** loss spike,
because the sign pattern of MLM representations captures very little retrieval
signal. This strong gradient signal forces the model to restructure its
embedding space in a binary-compatible way from the ground up. Standard
fine-tuning without this constraint restructures toward float-space retrieval,
and applying `sign()` post-hoc then discards fine-grained information the
model relied on.
Result: BAT > post-hoc binarization.

**Why BAT/annealed tanh on RoBERTa also beats the unquantized fine-tuned
version:** The binary constraint acts as regularization — it prevents the
model from relying on fine-grained magnitude differences that do not
generalize out-of-domain. This is supported by
[(Camuto et al., QReg, 2022)](https://arxiv.org/pdf/2206.12372) and
[(Sokolova et al., QT-DoG, 2024)](https://arxiv.org/html/2410.06020), who
show that quantization-aware training converges to flatter loss minima that
generalize better out-of-distribution — which is exactly what NanoBEIR tests.



- wait does that mean as i increas the epoch of training, the finetuned version of roberta becomes better and better and catching up with the performance of BAT , annealed tanh. In this case if finetuned version is getting better post ad hoc binarization also becomes better. and the post hoc binarizationa nd bat will converge? 

## What is NanoBEIR?

NanoBEIR is a heavily downsampled version of the BEIR benchmark
[(Thakur et al., 2021)](https://arxiv.org/abs/2104.08663), created by the
sentence-transformers team (UKPLab / Hugging Face) and released as part of
sentence-transformers v3.3.0. It is not backed by its own peer-reviewed paper
— it is an engineering artifact designed for fast evaluation.

## Why does it exist?

Full BEIR evaluation is expensive: corpora range from thousands to millions of
documents, and encoding them all at every checkpoint is prohibitive. NanoBEIR
provides a cheap proxy — fast enough to run during or after training — while
covering the same diverse set of retrieval domains as BEIR.

## How was it subsetted?

Each BEIR dataset is downsampled to approximately **50 queries** and a small
corpus (e.g., NanoQuoraRetrieval: 50 queries, ~5,000 documents). The 13
datasets covered are: `climatefever`, `dbpedia`, `fever`, `fiqa2018`,
`hotpotqa`, `msmarco`, `nfcorpus`, `nq`, `quoraretrieval`, `scidocs`,
`arguana`, `scifact`, and `touche2020`. There is no published formal
justification for how queries were selected — it appears to be random
subsampling.

## Can you use NanoBEIR in an academic setting instead of BEIR?

**Defensible, but with caveats you should acknowledge explicitly.**

**In favour:**
- Widely adopted in practice — used on the Hugging Face MTEB leaderboard and
  in many recent embedding model papers as a fast evaluation proxy.
- Covers the same domain diversity as BEIR, so relative rankings between
  models are generally preserved.

**Against:**
- No peer-reviewed paper formally validating the subsampling strategy or its
  correlation with full BEIR results.
- With only ~50 queries per subset, statistical power is low — significance
  tests (like your Wilcoxon) are more likely to return non-significant results
  even for real differences. This is a meaningful limitation to state.
- Full BEIR has known validity issues already
  [(Thakur et al., 2021)](https://arxiv.org/abs/2104.08663); NanoBEIR
  inherits them without adding any corrective measures.

**Recommended framing for your thesis:** Use NanoBEIR explicitly as an
*efficient proxy* for BEIR-style zero-shot evaluation, cite the
sentence-transformers collection on Hugging Face, and note the small query
count as a limitation on statistical power. If any key result is borderline,
validate it on full BEIR subsets.


- Here is a finding i found a bit olld and dont know why it is ta tway it is.
I took base bge model and trained/fine tuned 3 different model on my training data (you may check the training data its a 4.3M datapoints). The 3 fine tuning expirements were unquantized model fine tuning, Binary aware training (with step function layer at the end with STE for backprop) and annealed tahnh instead of STE. Also the unquantized model embeddings were post ad hoc quantised/ binarized. Which means there are 4 embedding results to comares using nanobeir benchmark. What i found was that, postad hoc binarized, BAT and annealead tahn embeddings all had simialr performance (no significance difference with wilcox test). When i ust swap the bge base model with roberta base model. What i saw was the BAt and annealed tanh had better formance than post ad hoc binarization and even the unquantised fine tuning version. How is this possible? Is there any explanation ? What expirments can i do to check to conclude one or the other technique is better ? how can i prove the bat or annelaed tanh is betetr? Although them being better than unquantised version is a bit shocking

Answer: 
Train something in between like 
RoBERTa-base → contriever → E5-base → BGE-base
(MLM only)    (unsup.)    (weak sup.) (fully specialized)

A. The "Information Bottleneck" Hypothesis
The Theory: The binary constraint is preventing overfitting on your 4.3M dataset.

The Experiment: Take a subset of your training data (e.g., 10%, 25%, 50%). If the unquantized model's performance drops significantly faster than the BAT model as data increases, you’ve proven that the binary model is a more robust regularizer.

B. Manifold Analysis (The "Why RoBERTa but not BGE?" question)
The Theory: BGE is too "rigid" to be reshaped, while RoBERTa is "plastic."

The Experiment: Use CKA (Centered Kernel Alignment) or Procrustes Analysis to measure how much the internal representations change during fine-tuning.

The Goal: Show that RoBERTa’s internal "map" changes drastically to accommodate bits, whereas BGE’s map resists change, leading to sub-optimal binary mapping.

A Master's thesis is rarely about discovering a "perfect" algorithm; it is about demonstrating a deep understanding of why things happen.

You are in a fantastic position because you have a counter-intuitive result (the binary model beating the unquantized one). In the eyes of a thesis committee, "Why did this happen?" is much more valuable than "I made a model that works."

Here is a step-by-step roadmap to turning these results into a high-quality thesis.

1. Pivot the Thesis Title/Theme
Don't just call it "Binarizing Embeddings." Call it something like:

"The Regularization Effects of Binary-Aware Training in Dense Retrieval: A Comparative Analysis of Pre-trained Manifold Plasticity."

This shifts the focus from "I tried some stuff" to "I am investigating a specific phenomenon."

2. The "Deep Dive" Methodology
To satisfy the requirements of a Master’s thesis, you should perform Ablation Studies and Diagnostic Tests. This is how you "dig into" the why:

A. The "Information Bottleneck" Hypothesis
The Theory: The binary constraint is preventing overfitting on your 4.3M dataset.

The Experiment: Take a subset of your training data (e.g., 10%, 25%, 50%). If the unquantized model's performance drops significantly faster than the BAT model as data increases, you’ve proven that the binary model is a more robust regularizer.

B. Manifold Analysis (The "Why RoBERTa but not BGE?" question)
The Theory: BGE is too "rigid" to be reshaped, while RoBERTa is "plastic."

The Experiment: Use CKA (Centered Kernel Alignment) or Procrustes Analysis to measure how much the internal representations change during fine-tuning.

The Goal: Show that RoBERTa’s internal "map" changes drastically to accommodate bits, whereas BGE’s map resists change, leading to sub-optimal binary mapping.

C. Visualizing the "Bit Utility"
The Experiment: Plot the distribution of your 768 dimensions. Are they balanced (50% zeros, 50% ones)?

The Goal: Calculate the Bit Entropy. If BAT has higher entropy than post-hoc binarization, you have mathematical proof that BAT "utilizes the binary space" more efficiently.

3. How to Structure the "Results & Discussion" Chapter
This is where most students struggle. Don't just list tables. Use this structure:

The Observation: "Unexpectedly, BAT/Annealed Tanh outperformed the unquantized baseline on RoBERTa-base."

The Hypothesis: "We hypothesize this is due to the 'quantization-as-regularization' effect, where the bit-constraint filters out low-magnitude noise in the 4.3M dataset."

The Proof: Insert your Bit Entropy or CKA charts here.

The Refinement: "However, this effect disappears with BGE, suggesting that the benefits of BAT are dependent on the initial state of the model's manifold."

4. Immediate Next Steps for You
Since you've already run the big 4.3M training runs (the hardest part!), the rest is "light" analysis:

Step 1: Run a Wilcoxon Signed-Rank Test across all NanoBEIR sub-tasks to show "Statistical Significance." Thesis advisors love p-values.

Step 2: Create a Visualization. A T-SNE plot showing the "Clustering" of RoBERTa-BAT vs. RoBERTa-Unquantized will be the centerpiece of your defense presentation.

Step 3: Draft the "Threats to Validity" section. Acknowledge that while BAT is better here, it might not be for every dataset size.



- can we talk about mnrl loss ? i heard it uses conise similarlity. The magnitude part is ignored and it helps learn directionally ony? thats weird. Then wy do we have float 32 embedding if its only directinal?
