# References

## Dense Retrieval Foundations

| Name | Venue | Key Information |
|------|-------|----------------|
| **DPR** — Karpukhin et al. https://arxiv.org/abs/2004.04906 | EMNLP 2020 | Canonical dual-encoder (bi-encoder) framework for retrieval. Encodes queries and passages independently into float32 vectors; the full-precision baseline that compression methods seek to replace. |
| **Sentence-BERT** — Reimers & Gurevych. https://arxiv.org/abs/1908.10084 | EMNLP 2019 | Introduces siamese fine-tuning of BERT for sentence embeddings. Defines the sentence-transformers library and MultipleNegativesRankingLoss used throughout this project. |
| **BGE / C-Pack** — Xiao et al. https://arxiv.org/abs/2309.07597 | SIGIR 2024 | Describes the BAAI/bge-base-en-v1.5 backbone fine-tuned in all experiments. Details the contrastive pre-training recipe and embedding geometry the binarization layers must preserve. |
| **ColBERT** — Khattab & Zaharia. https://arxiv.org/abs/2004.12832 | SIGIR 2020 | Late-interaction retrieval: stores per-token embeddings and scores at query time. Represents the accuracy-efficiency opposite of single-vector compression; useful contrast for motivating the bi-encoder + compression approach. |
| **SPLADE** — Formal et al. https://arxiv.org/abs/2107.05720 | SIGIR 2021 | Sparse neural retrieval via BERT + ReLU activation. Strongest sparse baseline on BEIR; uses inverted indexes without binary vector compression, clarifying the trade-off space. |
| **TAS-B** — Hofstätter et al. https://arxiv.org/abs/2104.06967 | SIGIR 2021 | Knowledge distillation from cross-encoders into bi-encoders (margin-MSE loss). Strong quality ceiling for dense retrieval; represents embeddings before any compression is applied. |
| **E5** — Wang et al. https://arxiv.org/abs/2212.03533 | arXiv 2022 | Weakly supervised contrastive pre-training for text embeddings. Achieves top MTEB/BEIR scores with the same BGE-style architecture; provides context for what large-scale pre-training adds to the embedding geometry. |

## Benchmarks

| Name | Venue | Key Information |
|------|-------|----------------|
| **BEIR** — Thakur et al. https://arxiv.org/abs/2104.08663 | NeurIPS 2021 | Heterogeneous zero-shot IR benchmark across 18 datasets. The NanoBEIR evaluation used in this project is a direct subset; all result tables are comparable to BEIR numbers from prior work. |
| **MTEB** — Muennighoff et al. https://arxiv.org/abs/2210.07316 | EACL 2023 | Covers 58 datasets across 8 embedding tasks including retrieval. Provides community-wide ranking context for compressed models relative to state of the art. |

## Matryoshka Representation Learning

| Name | Venue | Key Information |
|------|-------|----------------|
| **MRL** — Kusupati et al. https://arxiv.org/abs/2205.13147 | NeurIPS 2022 | Original MRL paper: trains embeddings jointly across nested sub-dimensions so any prefix is independently useful. Directly implemented in e3/e4 over [32, 64, 128, 256, 512] dims using MatryoshkaLoss. |
| **QAMA** — https://dl.acm.org/doi/10.1145/3746252.3761077 | CIKM 2025 | Closest prior work to this thesis: combines MRL with quantization-aware training for retrieval embeddings. Retains 96% nDCG@10 at 192 dims + 2-bit. Uses a penalty loss rather than a soft differentiable staircase — the gap this work fills. |

## Binarization / Semantic Hashing for Retrieval

| Name | Venue | Key Information |
|------|-------|----------------|
| **BPR** — Yamada et al. https://arxiv.org/abs/2106.00882 | ACL 2021 | Binarizes DPR passage embeddings via learning-to-hash; 32× memory reduction with near-zero accuracy loss. Closest NLP/IR prior work; the annealed tanh approach is a cleaner end-to-end differentiable alternative. |
| **Semantic Hashing** — Salakhutdinov & Hinton. https://www.cs.utoronto.ca/~rsalakhu/papers/semantic_final.pdf | IJAR 2009 | Original neural semantic hashing: maps documents to binary Hamming codes via a deep model. Conceptual ancestor of all binary embedding work; the annealed tanh binarization is its modern differentiable descendant. |
| **BEBR** — Gan et al. https://arxiv.org/abs/2302.08714 | KDD 2023 | Industrial-scale multi-bit binary retrieval at Tencent using residual binarization; 30–50% index cost reduction. Closely related to the multi-bit staircase quantizer direction (e6). |
| **DBQ** — Liu et al. https://ojs.aaai.org/index.php/AAAI/article/view/8208 | AAAI 2012 | Foundational 2-bit hashing paper. Key insight: a single threshold at 0 sits at peak embedding density, causing instability. Two symmetric thresholds around 0 fix this — direct precursor to multi-bit threshold design. |
| **Multi-Bit Hashing** — Weng et al. https://openaccess.the

cvf.com/content/ACCV2022/papers/ | ACCV 2022 | Adaptive per-dimension threshold placement for retrieval; thresholds are placed to match each projection's empirical distribution. Directly addresses where to place thresholds when embedding range is unknown. |
| **Binary MRL** — Mixedbread. https://mixedbread.com/blog/binary-mrl | Blog 2024 | Joint MRL + binary quantization training (same combination as e4/e5). Shows that training with binarization in mind — not post-hoc — is critical for retrieval quality. |
| **HF Embedding Quantization** — HuggingFace. https://huggingface.co/blog/embedding-quantization | Blog 2024 | Standard reference for 1-bit binary (sign at 0) and int8 scalar quantization in sentence-transformers. The baseline approach this thesis improves on. |

## Differentiable Quantization

| Name | Venue | Key Information |
|------|-------|----------------|
| **STE** — Bengio et al. https://arxiv.org/abs/1308.3432 | arXiv 2013 | Introduces the Straight-Through Estimator: copy the gradient through a non-differentiable sign(x). Primary reference for the STE binarization layer used in e2 and e4 — the direct baseline the annealed tanh improves upon. |
| **BNN** — Courbariaux et al. https://arxiv.org/abs/1602.02830 | NeurIPS 2016 | Canonical application of STE to binarize weights and activations in vision models. Universally cited when introducing any binarization layer in a neural network. |
| **Quantization Networks** — Yang et al. https://arxiv.org/abs/1911.09464 | CVPR 2019 | Canonical sum-of-sigmoids quantizer with learnable thresholds and annealed temperature. Represents a k-level quantizer as k-1 weighted sigmoid functions that sharpen into step functions. The architecture e5 directly generalises. |
| **Trainable Bitwise Soft Quantization** — https://arxiv.org/abs/2603.05172 | arXiv 2025 | Most directly analogous setup: sigmoid staircase applied end-to-end on input features with temperature annealing. Achieves 5–16× compression with modest accuracy loss. |
| **LSQ** — Esser et al. https://arxiv.org/abs/1902.08153 | ICLR 2020 | Learns a single scale (step size) per layer with an STE gradient. Dominant baseline for differentiating through uniform quantization; technique is directly applicable to the threshold training in e6. |
| **QIL** — Jung et al. https://arxiv.org/abs/1808.05779 | CVPR 2019 | Learns actual interval boundaries (thresholds) by backpropagating task loss. Direct ancestor of the anchor + softplus-gaps threshold parameterisation used in the multi-bit staircase layer. |
| **Amazon temp. compression** — https://openreview.net/forum?id=szRmEM8Kx5 | ICLR 2025 | Shows contrastive training temperature directly controls post-hoc compressibility of embeddings, reducing intrinsic dimensionality. Complementary evidence that training-time choices (like annealing schedule) determine compression effectiveness. |

## Vector Compression & ANN Search

| Name | Venue | Key Information |
|------|-------|----------------|
| **PQ** — Jégou et al. https://dl.acm.org/doi/10.1109/TPAMI.2010.57 | IEEE TPAMI 2011 | Defines Product Quantization: split embedding into sub-vectors, quantize each independently. Directly implemented as a post-hoc transform in eval_v2/wrappers/transforms.py; required citation when PQ results appear in tables. |
| **FAISS** — Johnson et al. https://arxiv.org/abs/1702.08734 | IEEE Trans. Big Data 2019 | Standard GPU-accelerated ANN library for billion-scale similarity search. Motivates why embedding compression matters: FAISS is the production system that binarized/PQ embeddings feed into. |
| **ScaNN** — Guo et al. https://research.google/blog/announcing-scann-efficient-vector-similarity-search/ | ICML 2020 | Anisotropic vector quantization: quantization error parallel to the query direction hurts MIPS more than orthogonal error. Theoretical backing for why cosine-preserving quantization (not just L2 reconstruction) is the right objective. |

## Backbone Models

| Name | Venue | Key Information |
|------|-------|----------------|
| **RoBERTa** — Liu et al. https://arxiv.org/abs/1907.11692 | arXiv 2019 | Robustly optimised BERT pre-training. One of the three backbones in these experiments; needed to explain pre-training methodology and why RoBERTa may behave differently from BGE under binarization. |



---

## Primary papers

Thease are the primary literature review papers, that recent

- CoRECT: A Framework for Evaluating Embedding Compression Techniques at Scale
- QAMA: Quantization Aware Matryoshka Adaptation
- Efficient Passage Retrieval with Hashing for Open-domain Question Answering (BPR) (introduced tanh)
- TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate
- Binary Embedding-Based Retrieval at Tencent (BEBR)
- SMEC:Rethinking Matryoshka Representation Learning for Retrieval Embedding Compression




In background section
- Explain the origin, then state how you use it

In Methodology section
- "Following X" / "Adapting X" / "Building on X"

In Related Work section 
- Position, contrast, differentiate
- example
"Concurrent with this work, CoRECT [cite] provides a controlled evaluation framework for embedding compression at scale, benchmarking eight post-hoc techniques on corpora up to 100M passages. However, CoRECT evaluates only post-hoc methods and does not study training-time compression or cross-axis combinations. This thesis extends the controlled-comparison approach to include training-time methods (STE, annealed tanh, MRL) and systematically evaluates their interactions."
"QAMA [cite] combines MRL with multi-level quantization-aware training for retrieval embeddings, achieving 96% of full-precision nDCG@10 at 192 dimensions with 2-bit quantization. Their approach uses a penalty loss to encourage quantization-friendly representations, whereas this thesis employs a smooth differentiable surrogate (annealed tanh) that progressively sharpens toward the target discrete function during training."

----

# paper notes i read for literature review

# CoRECT: A Framework for Evaluating Embedding Compression Techniques at Scale (Jan 2026)
- paper: https://arxiv.org/abs/2510.19340
- **Current Gap and its main focus:** While embedding compression methods solve this index-size issue, prior evaluation studies often overlook the critical role of **corpus complexity** (such as corpus size and document length), which strongly affects dense retrieval performance.

- Contribution
	- **CoRECT** (Controlled Retrieval Evaluation of Compression Techniques)
		- systematic, large-scale evaluation of embedding compression methods
	- **CoRE** (Controlled Retrieval Evaluation) dataset, curated benchmark from MS MARCO v2 with controlled corpus sizes (10K → 100M passages, 10K → 10M documents), using 76 human-judged TREC DL queries
	- **Benchmark results** — 4 models × 8+ compression types × 40+ configurations
	- Models (4 of them)

| Model                 | Params | Dims | MRL-trained?   |
| --------------------- | ------ | ---- | -------------- |
| Jina V3               | 572M   | 1024 | ✓ (6 cutoffs)  |
| Multilingual E5 Large | 560M   | 1024 | ✗              |
| Snowflake-M V2        | 305M   | 768  | ✓ (cutoff 256) |
| Snowflake-M V1        | 109M   | 768  | ✗              |
- 8 compression techniques 
	- floating point casting: FP16, BF16, FP8 (two variants)
	- Scalar & Binary Quantization (SBQ): uint8/4/2-bit (equal-distance or percentile binning); binary (zero or median threshold)
	- Dimensionality Reduction: MRL truncation, PCA
	- Hashing / Learning-to-Hash: LSH, Product Quantization (PQ)
	- MRL + quantisation was also done to achieve higher compression ratios
- Findings
	- Corpus complexity significantly hurts retrieval
	- Document retrieval degrades faster than passage retrieval
	- Quantization > vector truncation (at equal compression ratio)
	- MRL training matters a lot for truncation
	- Combining techniques often wins: For 3 of 4 models, **quantization + moderate truncation** achieved the best recall at a fixed compression ratio
	- Wrong compression method can seriously hurt performance
	- Metric choice matters: **NDCG@10** (single-stage) stays surprisingly stable — random documents rarely break into top-10 even at 100M scale

#  QAMA: Quantization Aware Matryoshka Adaptation (Nov 2025)
- paper: https://dl.acm.org/doi/epdf/10.1145/3746252.3761077
#### **Overview**

- **Goal:** Optimize embedding storage and retrieval speed for Information Retrieval (IR) systems.
    
- **Context/Problem:** While MRL creates flexible embeddings, the paper states that MRL severely struggles when combined with aggressive compression/quantization techniques.
    
- **Solution:** **QAMA**. A framework that uses Transformer models (ModernBERT, MiniLM) augmented with Feed-Forward Networks (FFN) and a hybrid quantization scheme.
    
- **Claim:** Achieves >90% storage savings while retaining 95–98% of the original full-precision (FP32) retrieval accuracy.
    

#### **Core Innovations**

1. **Hierarchical Embeddings (MRL):** Enhances the semantic embedding model with a nested hierarchy _(Note: As you correctly observed, this builds on existing MRL literature and is less of an architectural novelty)._
    
2. **Trainable Multi-Level Quantization:** Employs learnable quantization thresholds that map vectors down to ultra-low bit levels (0.5-bit to 2-bit).
    
3. **Hybrid Precision Architecture (Non-Uniform Quantization):** Because MRL front-loads important data, this architecture adaptively assigns **higher bit precision to early dimensions** and **lower bit precision to later dimensions**.
    
4. **Multi-Level Bit Expansion & Fast Retrieval:** Extends standard Hamming distance to multi-level quantized embeddings via a clever bit-level expansion. This enables highly efficient similarity searches using native hardware bitwise operations.
    

#### **Methodology & Architecture**

**1. Quantization Levels & Thresholds**

- Supports multiple granularities:
    
    - **2-bit:** 4 distinct levels.
        
    - **1.5-bit:** 3 distinct levels.
        
    - **1-bit:** 2 distinct levels.
        
    - **0.5-bit:** Compresses heavily by combining pairs of dimensions.
        
- **Learnable Thresholds:** For $k$-bit quantization ($2^k$ levels), the model requires $2^k - 1$ learnable thresholds.
    
- **Initialization & Training:** Thresholds are initialized using dataset percentile statistics and updated during training via an Exponential Moving Average (EMA).
    

**2. The Hybrid Quantization Scheme**

The vector is sliced into quarters, progressively aggressively quantizing the less important MRL dimensions:

- **First 25%:** 2-bit
    
- **Next 25%:** 1.5-bit
    
- **Next 25%:** 1-bit
    
- **Final 25%:** 0.5-bit (Requires combining bits using the FFN)
    
- **Net Result:** An average of just **1.625 bits per dimension**.
    

**3. Efficient Storage & Similarity Computation**

- Quantized vectors are mapped into binary form, padded if needed, and packed tightly into `uint64` arrays.
    
- **Binary Codebook Mapping:** Uses a system mathematically identical to unary/Hamming codes to ensure distances remain proportional.
    
    - _2-bit to 3-bit mapping:_ `0 -> [0,0,0]`, `1 -> [0,0,1]`, `2 -> [0,1,1]`, `3 -> [1,1,1]`
        
    - _1.5-bit to 2-bit mapping:_ `0 -> [0,0]`, `1 -> [0,1]`, `2 -> [1,1]`
        
- **Fast Distance Calculation:** Because of this codebook mapping, standard cosine similarity can be entirely replaced by **Hamming Similarity**, which is executed incredibly fast at the CPU level using `XOR` and `POPCOUNT` operations.
    

#### **Evaluation & Results**

- **Benchmark:** Tested on the MTEB benchmark suite (using 15 BEIR-like subsets).
    
- **Baselines for Comparison:**
    
    - Original full-precision model (FP32, FP16, INT8).
        
    - Simple, non-learned threshold quantization without MRL.
        
    - An upper-bound baseline: A non-MRL model fine-tuned specifically to match output dimensions using an FFN.
        
- **Performance (NDCG@10):**
    
    - **2-bit Quantization:** Recovers 95% to 98% of the FP32 model's original performance.
        
    - **Hybrid Quantization:** Maintains the impressive accuracy of the 2-bit setup while yielding further storage savings (dropping to ~1.625 bits/dim).
        
- **Reproducibility:** _The authors have not currently shared a public code repository, making exact reproduction difficult._




# Efficient Passage Retrieval with Hashing for Open-Domain QA (BPR June 2021)
- paper: https://arxiv.org/abs/2106.00882
**Overview**

- **Core Contribution:** Introduces Binary Passage Retriever (BPR), integrating learning-to-hash (different one) into Dense Passage Retriever (DPR) for highly efficient, low-memory RAG pipelines.
    
- **Impact:** Compresses index memory from 65 GB to 2 GB (~32x reduction) with zero accuracy loss on Natural Questions (NQ) and TriviaQA. Outperforms classical BM25 and standard quantization baselines.
    

**Architecture & 2-Stage Retrieval**

BPR utilizes a two-stage approach: fast filtering via bitwise operations, followed by precision reranking.

Plaintext

```
[ Query / Question ] 
        │
        ▼
[ Encoder + Hash Layer ] ──(Continuous to Binary Layer)──┐
        │                                                │
        ├──> Continuous Vector (e_q)                     │
        │                                                ▼
        │                                 [ Binary Hash Code (h_q) ]
        │                                                │
        │                                                ▼
        │                         [ Stage 1: Candidate Generation ]
        │                       (Hamming Distance on Binary Index)
        │                                                │
        │                                                ▼
        │                             [ Top-K Candidate Passages ]
        │                                                │
        └──────────────────────────────────────────────> ▼
                                              [ Stage 2: Reranking ]
                               (Inner Product: e_q • e_candidate_continuous)
                                                         │
                                                         ▼
                                                [ Final Top Passages ]
```

**Methodology & Training**

- **End-to-End Training:** Simultaneously trains the continuous encoder and the binary hash function (multi-task learning).
    
- **Differentiability:** The standard `sign()` function cannot be backpropagated. BPR approximates it utilizing a scaled `tanh` function with temperature annealing during training.
    
- **Distinction:** Unlike Product Quantization (PQ) which acts as a post-training compression step, this learning-to-hash approach is an active layer trained directly for the QA task.
    

**Loss Function**

Training minimizes a joint objective: $\mathcal{L}_{total} = \mathcal{L}_{cand} + \mathcal{L}_{rerank}$

1. **Candidate Loss ($\mathcal{L}_{cand}$):** Margin ranking loss utilizing the approximated continuous hash codes ($\tilde{h}$). Inner product and Hamming distance are functionally interchangeable here.
    
    $$\mathcal{L}_{cand} = \sum_{j=1}^{n}\max(0, -(\langle\tilde{h}_{q_{i}},\tilde{h}_{p_{i}^{+}}\rangle - \langle\tilde{h}_{q_{i}},\tilde{h}_{p_{i,j}^{-}}\rangle) + \alpha)$$
    
    - $\tilde{h}_{q_{i}}$: Question's approximated hash code.
        
    - $\tilde{h}_{p_{i}^{+}}$: Relevant passage's approximated hash code.
        
    - $\tilde{h}_{p_{i,j}^{-}}$: Irrelevant passage's approximated hash code.
        
2. **Reranking Loss ($\mathcal{L}_{rerank}$):** Standard Multi-Negative Ranking Loss (MNRL) applied to the continuous embeddings ($e_q$, $e_p$).
    

**Experiments & Results**

- **Baselines:** DPR (Linear Scan), DPR + HNSW, DPR + LSH, DPR + PQ. All configured to match BPR's bits-per-passage constraint for a fair memory footprint comparison.
    
- **Results:**
    
    - **Accuracy:** Matches the Top-20 and Top-100 retrieval accuracy of the uncompressed DPR (Linear Scan).
        
    - **Efficiency:** Drastically outperforms DPR + LSH and DPR + PQ in retrieval accuracy at the same bit rate constraint.
        
    - **Speed vs. Memory:** Generates candidates significantly faster than linear scans and avoids the massive RAM overhead required by graph-based indices like HNSW.



# BEBR: Binary Embedding-based Retrieval at Tencent (Feb 2023)
- paper: https://arxiv.org/abs/2302.08714

## Overview & Motivation
* **Context:** Embedding-based retrieval (EBR) drives modern web search, social search, and e-commerce search, typically utilizing a "recall-rerank" architecture.
* **The Bottleneck:** The first-stage recall module suffers from high latency, memory, and computation costs due to the massive scale of document corpora.
* **Limitations of Current ANN:** Advanced Approximate Nearest Neighbor (ANN) algorithms (like HNSW) reduce computation but require elaborate, expensive engineering to plug into existing industrial systems.
* **BEBR Value Proposition:**
    * Compresses full-precision float vectors into customized binary vectors.
    * Task-agnostic, model-agnostic, and natively supports multi-modality.
    * Seamless integration into existing ANN algorithms (IVF, HNSW) without structural changes.
    * Achieves 30%–50% cost savings (index memory and disk usage) with near-lossless accuracy.

## Architecture: Recurrent Binarization
Instead of using a simple step function for a one-time forward pass, BEBR utilizes a recurrent lightweight transformation model with non-linear (ReLU) MLP blocks. It progressively refines the representation to capture more information.

Let the initial full-precision float embedding be $x_0$.

* **Binarization Block:** Transforms the continuous vector into a binary vector $b_t$.
  $$b_t = \text{sign}(\text{MLP}_{bin}(x_{t-1}))$$
  *Note:* Because the sign function is non-differentiable, the Straight-Through Estimator (STE) is utilized to approximate gradients during backpropagation.

* **Reconstruction Block:** Attempts to recover the original float representation from the compressed binary bits.
  $$\hat{x}_t = \text{MLP}_{rec}(b_t)$$

* **Residual Block:** Computes the difference (residual) between the input and the reconstructed vector. This residual is passed to the next loop.
  $$x_t = x_{t-1} - \hat{x}_t$$

* **Customizable Compression:** This loop repeats $u+1$ times. The final binary representation is the concatenation of all binary outputs ($b_1, b_2, \dots, b_{u+1}$). The total bit length is $m \times (u+1)$, where $m$ is the MLP output dimension, allowing you to tailor the number of bits to balance accuracy and cost.

## Task-Agnostic Training
* **Decoupled Architecture:** The binarization module is trained independently from the heavy backbone networks (CNNs, Transformers). It only requires about 2 V100 GPU hours for millions of vectors.
* **Training Data:** It utilizes pre-extracted floating-point vectors as input rather than raw images or text. This makes the training entirely agnostic to the task or modality.
* **Objective Function:** It uses a contrastive learning objective (e.g., InfoNCE) equipped with queue-based hard negative mining. The loss function optimizes the float-to-binary network by clustering positive embedding pairs and pushing apart negative pairs directly in the binary space.

## Compatibility of Deep Neural Networks (Backward Compatibility)
* **The Problem:** Traditional model upgrades require "backfilling"—re-computing the embeddings for the entire database of billions of documents to refresh the index. This is computationally expensive and takes weeks.
* **The Solution (BCT):** Backward Compatible Training (BCT) enables backfill-free upgrades. The new binarization model is trained asymmetrically so that queries encoded by the new model map correctly to the gallery embeddings indexed by the old model.
* **Deployment Impact:** A new, more accurate embedding model can be instantly deployed to process incoming user queries and successfully search against the older, un-refreshed database index.

## Symmetric Distance Calculation (SDC) & SIMD
* **Standard Approach:** Binary vectors are typically compared using Hamming distance via basic XOR and POPCNT instructions.
* **SDC Mechanism:** SDC achieves lower response latency than Hamming codes by computing distance look-up tables.
* **How SIMD Works Here:** SDC heavily exploits Single Instruction, Multiple Data (SIMD) units found in modern computer architectures. By loading the pre-computed distance tables directly into the CPU's in-register cache (utilizing instructions like AVX), SIMD allows the CPU to execute parallel look-up instructions across multiple data points in a single clock cycle. This bypasses memory bandwidth bottlenecks and calculates the distance for multiple binary segments simultaneously.

## Evaluation
* **Offline Benchmark:** Evaluated using Recall@K (e.g., Recall@1, @5, @10). Tested on the public MS-COCO Caption dataset, as well as massive industrial datasets collected from Tencent's web search and video copyright pipelines.
* **Online A/B Test:** Deployed live across several Tencent products (Sogou, Tencent Video, QQ World). System performance and search satisfaction were evaluated using Click-Through Rate (CTR) and Query Rewrite Rate.



# SMEC: Rethinking Matryoshka Representation Learning for Retrieval Embedding Compression (Oct 2025)

**Source:** [https://arxiv.org/pdf/2510.12474](https://arxiv.org/pdf/2510.12474)

## Overview

Standard **Matryoshka Representation Learning (MRL)** encodes hierarchical information into a single embedding, allowing progressive truncation for computational efficiency. While MRL has inspired variants like the _Matryoshka-Adaptor_ to map embeddings into structured representations, it struggles with sub-optimal convergence because abruptly truncating to lower dimensions (without explicit training for them) degrades performance.

**SMEC (Sequential Matryoshka Embedding Compression)** introduces a progressive training framework that resolves three core limitations of MRL.

## 1. The Three Limitations of MRL & SMEC's Solutions

### Limitation 1: Gradient Fluctuation

- **The Cause (The Multi-Objective "Tug-of-War"):** During backpropagation, jointly optimizing all dimensions creates conflicting gradients on shared weights. This leads to high variance, unstable learning, and suboptimal convergence.
    
- **SMEC's Solution (Sequential MRL):** Replaces simultaneous joint optimization with a step-by-step sequential training framework to stabilize gradients.
    

### Limitation 2: Information Degradation

- **The Cause:** Blindly slicing off the ends of an embedding (truncation) destroys nuanced semantic details.
    
- **SMEC's Solution (Adaptive Dimension Selection - ADS):** An intelligent mechanism to evaluate and preserve critical features during dimension pruning.
    

### Limitation 3: Weak Sample Selection

- **The Cause:** MRL relies on in-batch negative samples, which are often too "easy," limiting how well the model learns.
    
- **SMEC's Solution (Selectable Cross-Batch Memory - S-XBM):** Improves unsupervised learning by finding "hard negatives." It calculates the similarity between a query and a global memory bank, fetching the top-K most similar results. These are computed _before_ training to ensure the embeddings remain consistent.
    

## 2. How Sequential MRL (SMRL) Works

Instead of jointly training all dimensions at once, SMRL trains them progressively and freezes prior states. _(Note on Weight Shape: The paper utilizes $N$ to represent the dimension/weight shape, though some diagrams mistakenly label it $M$.)_

### Step 1: Base Training

- The embedding passes through an $(N \times N)$ Fully Connected (FC) weight vector.
    
- **Compute Loss:** $L_1(x) = L(x[1:N]) + L(x[1:N/2])$
    
- Backpropagate and update the $(N \times N)$ weights.
    
- Freeze these weights before moving to the next step.
    

### Step 2: First Truncation

- The full-dimension output from Step 1 is passed through a sub-FC layer of size $(N \times N/2)$.
    
- **Compute Loss:** $L_2(x) = L(x[1:N/2]) + L(x[1:N/4])$
    
- Backpropagate to update only this new sub-FC layer.
    

### Step 3 & Beyond

- This sequential process continues, systematically compressing the embedding down to the target lower dimensions without gradient interference from the larger dimensions.
    

## 3. Experimental Results

- **Datasets Evaluated:** Tested across image, text, and multimodal datasets, specifically highlighting **BEIR**, **Fashion-200k**, and **Products-10k**.
    
- **Metric Used:** NDCG@10.
    
- **Base Models Tested:** OpenAI text embeddings and the open-source LLM2Vec.
    
- **Key Achievements:** At **128 dimensions** on the BEIR dataset, SMEC improved performance by **1.9 points** (OpenAI embeddings) and **1.1 points** (LLM2Vec embeddings) compared to the Matryoshka-Adaptor baseline.






# TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (April 2025)
paper: https://arxiv.org/abs/2504.19874
## What Problem Does This Solve?

Large language models store **KV (key-value) cache** during inference — vectors for every token, every layer, every attention head. This grows huge fast, bottlenecking memory and speed.

**Vector quantization (VQ)** compresses these high-dimensional floating-point vectors into low-bit integers. The challenge: do it with minimal distortion, fast enough for real-time inference, without needing to see the data first.

Existing methods fail on at least one of:

- **Distortion** — too far from theoretical optimum
- **Speed** — slow preprocessing, not GPU-friendly
- **Online use** — require training on data before they can quantize

TurboQuant solves all three.

---

## Background: Shannon's Rate-Distortion Theory

Shannon (1948) asked: **what is the absolute minimum distortion achievable at a given bit budget?**

The answer is the **distortion-rate function**:

```
D(R) = σ² · 2^(−2R)
```

For a Gaussian source with MSE distortion:

- Every extra bit per coordinate → distortion drops by **4×**
- This exponential `4^(-b)` pattern appears throughout the paper
- **No algorithm can beat this** — it's an information-theoretic wall

### Shannon Lower Bound (SLB)

For vectors on the unit hypersphere:

```
D(B) ≥ 4^(−B/d)
```

where B = total bits, d = dimension. This is TurboQuant's benchmark.

---

## Core Insight: Why Random Rotation?

### The problem with raw vectors

LLM embedding vectors have **unknown, correlated structure** across dimensions. To quantize well, you'd need a custom codebook designed for that specific data — expensive, offline, impractical.

### What rotation does

A random rotation **does not change geometry** — distances, inner products, and angles between any two vectors are perfectly preserved.

What it _does_ change is the **coordinate representation**:

```
Before rotation:             After rotation:
dim 1: clustered near 0.5    dim 1: ~Gaussian(0, 1/d)
dim 2: bimodal at ±3    →    dim 2: ~Gaussian(0, 1/d)
dim 3: skewed right          dim 3: ~Gaussian(0, 1/d)
...                          all same distribution!
```

After rotation, every coordinate independently follows a **Beta distribution** (converging to Gaussian as d grows). This is a mathematical consequence of uniform distribution on the hypersphere — proven in Lemma 1 of the paper.

### Why this enables efficient quantization

Because all coordinates now follow the **same known distribution**, you can:

1. Design **one optimal scalar quantizer** for that distribution
2. Apply it identically to all d coordinates of any vector
3. No data needed — works instantly on any input

### The same rotation for everyone

The rotation matrix R is sampled **once** and fixed forever. Every vector in the dataset gets multiplied by the same R. This is essential — otherwise inner products between vectors would be destroyed.

In practice, a **randomized Hadamard transform** is used instead of a dense matrix, reducing cost from O(d²) to O(d log d).

---

## Stage 1: MSE-Optimal Quantization via Lloyd-Max

### Goal

Minimize reconstruction error:

```
D_mse = E[ ||x − Q⁻¹(Q(x))||² ]
```

### Lloyd-Max Quantizer

Given a distribution and k = 2^b levels, Lloyd-Max alternates:

1. **Boundaries** → midpoint between adjacent levels
2. **Levels** → mean of the distribution within each bucket

This is 1D k-means on a continuous distribution. It converges to the placement that minimizes MSE. The key fact: **the mean of a bucket minimizes squared error** within that bucket (provable by calculus).

TurboQuant precomputes Lloyd-Max codebooks for Beta/Gaussian distributions at each useful bit-width (1–8 bits), then reuses them for every vector.

### Distortion guarantee (Theorem 1)

For any unit-norm vector x ∈ ℝ^d:

```
D_mse ≤ √(3π/2) · 4^(−b)  ≈  2.7 × (optimal)
```

At specific bit-widths:

|Bits (b)|TurboQuant MSE|Lower bound|
|---|---|---|
|1|0.36|~0.25|
|2|0.117|~0.063|
|3|0.030|~0.016|
|4|0.009|~0.004|

### Why only 2.7× from optimal?

The gap comes from doing **scalar** (per-coordinate) quantization instead of **joint** quantization across all dimensions. Joint quantization would be theoretically better but requires exponentially large codebooks — computationally infeasible. The near-independence of coordinates after rotation makes scalar quantization nearly as good.

---

## The MSE Problem: Why It Breaks Inner Products

### Lloyd-Max is biased

The mean minimizes squared error, but it **systematically shifts** values:

```
True values in bucket [0.3, 0.7]:   0.31,  0.45,  0.62,  0.69
Lloyd-Max maps all to:               0.52  (the mean)

Values below 0.52 → pulled UP
Values above 0.52 → pulled DOWN
```

The reconstruction error `x - x̂` is not zero-mean — it depends on where within the bucket the true value falls. This causes **bias in inner product estimation**:

```
E[<y, x̂>] ≠ <y, x>    ← biased!
```

Small at high bit-widths (tiny buckets), but significant at 2–3 bits.

### Why bias matters

- **Attention mechanisms**: softmax over biased scores → wrong token weights
- **Nearest neighbor search**: biased distances → wrong rankings
- Errors don't cancel — they accumulate predictably in one direction

---

## Stage 2: Unbiased Inner Product via QJL Residual

### The two-stage approach

Instead of fixing the MSE quantizer, apply a **residual correction**:

```
Stage 1: x̂₁ = MSE_quantize(x, bits = b−1)
          residual = x − x̂₁         ← minimized by MSE quantizer

Stage 2: x̂₂ = QJL(residual)         ← 1-bit unbiased correction

Final:    x̂  = x̂₁ + x̂₂
```

Total budget: (b−1) + 1 = b bits. ✓

### What is QJL?

Quantized Johnson-Lindenstrauss (QJL) is a 1-bit quantizer:

```
Quantize:     QJL(r) = sign(S · r)        S is random Gaussian matrix
Reconstruct:  Q⁻¹(z) = √(π/2)/d · Sᵀ · z
```

**Key property**: provably unbiased for inner products:

```
E[<y, Q⁻¹(QJL(r))>] = <y, r>    for any y
```

This follows from classical Johnson-Lindenstrauss theory — the sign of a Gaussian projection is an unbiased estimator of the inner product.

### Why does this make the full estimator unbiased?

```
E[<y, x̂₁ + x̂₂>]
= E[<y, x̂₁>] + E[<y, QJL(x − x̂₁)>]
= E[<y, x̂₁>] + <y, x − x̂₁>       ← QJL is unbiased on residual
= E[<y, x̂₁>] + <y, x> − <y, x̂₁>
= <y, x>                             ← bias cancels exactly ✓
```

The bias from stage 1 is exactly cancelled by the unbiased residual correction.

### Why (b−1) bits for stage 1?

Because QJL's variance is proportional to `||residual||²`. Stage 1 at (b−1) bits **minimizes the residual norm** — giving stage 2 the smallest possible residual to handle.

`(b−1) + 1` is the optimal split between the two stages.

---

## Information-Theoretic Lower Bounds (Theorem 3)

Using Shannon's Lower Bound + Yao's minimax principle, the paper proves that for **any** randomized quantizer with b bits per coordinate:

```
D_mse  ≥  4^(−b)                       (MSE lower bound)
D_prod ≥  ||y||² / d  ·  4^(−b)        (inner product lower bound)
```

These are hard limits. No algorithm can do better on worst-case inputs.

TurboQuant's gap to these bounds:

- MSE: factor of √(3π/2) ≈ **2.7×** — constant, independent of b and d
- At b=1: only **1.45×** from optimal

---

## Full Pipeline Summary

### Quantization

```
x  (raw 512-dim vector, float32)
 ↓  x' = R · x                    same R for all vectors, O(d log d)
x' (rotated — coords ~Gaussian, nearly independent)
 ↓  Lloyd-Max per coordinate       precomputed codebook lookup
q  (512 integers, b bits each)     stored compressed
```

### Dequantization (MSE)

```
q  → look up level values → x̂' → x̂ = Rᵀ · x̂'
```

### Dequantization (Inner product)

```
q₁ (b−1 bit MSE codes) + q₂ (1-bit QJL codes)
 → x̂₁ + x̂₂ = Rᵀ · (MSE_reconstruct(q₁) + QJL_reconstruct(q₂))
```

---

## Experimental Results

### KV Cache Compression

Tested on long-context LLM benchmarks:

|Bits/channel|Quality impact|
|---|---|
|16 bits (baseline)|—|
|3.5 bits|**Zero degradation**|
|2.5 bits|Marginal degradation|

That's **>5× compression** with essentially no quality loss.

Also achieves perfect retrieval on **needle-in-a-haystack** tasks (long-context memory benchmarks).

### Nearest Neighbor Search

- **Outperforms** data-dependent product quantization (PQ) in recall
- **Indexing time → ~zero** (no k-means training required)
- Works on streaming/online data with no preprocessing

---

## Comparison to Existing Methods

|Method|Online?|Near-optimal distortion?|GPU-friendly?|
|---|---|---|---|
|Product Quantization (PQ)|❌ needs k-means|❌|✅|
|Grid-based PQ|✅|❌ (loose bounds)|❌ slow search|
|Hessian-based (GPTQ etc.)|❌ heavy preprocessing|varies|partial|
|QJL alone|✅|✅ at 1-bit only|✅|
|**TurboQuant**|✅|✅ all bit-widths|✅|

---

## Key Takeaways

1. **Random rotation** makes coordinates identically distributed and nearly independent — enabling optimal scalar quantization without data
2. **Lloyd-Max** is optimal for MSE but introduces inner product bias
3. **Two-stage (MSE + QJL residual)** eliminates bias while staying near-optimal
4. **Within 2.7× of Shannon's lower bound** — provably, for any input, any bit-width
5. **Zero preprocessing** — critical for online applications like KV cache

The core elegance: instead of adapting the quantizer to the data (hard, offline), TurboQuant adapts the data to a fixed optimal quantizer (easy, online).

