## Chapter 1. Introduction

- 1.1 Motivation
- 1.2 Problem Statement
- 1.3 Research Questions
- 1.4 Contributions
- 1.5 Thesis Organization

---

## Chapter 2. Background and Related Works

- 2.1 Dense Text Retrieval
    - 2.1.1 Bi-Encoder Architecture
    - 2.1.2 Multiple Negatives Ranking Loss
- 2.2 Post-Hoc Embedding Compression
    - 2.2.1 Scalar Quantization (INT8, INT4)
    - 2.2.2 Binary Quantization and Hamming Distance
    - 2.2.3 Product Quantization
    - 2.2.4 TurboQuant
- 2.3 Training-Time Compression
    - 2.3.1 Straight-Through Estimator
    - 2.3.2 Binary-Aware Training
    - 2.3.3 Matryoshka Representation Learning
    - 2.3.4 Quantization-Aware Training in the Broader Literature
- 2.4 Evaluation Benchmarks
    - 2.4.1 BEIR
    - 2.4.2 NanoBEIR
    - 2.4.3 NASA Domain-Specific Benchmarks

---

## Chapter 3. Datasets

- 3.1 Training Data
    - 3.1.1 Source Datasets (MS MARCO, S2ORC, Quora, NLI-SimCSE, Natural Questions, TriviaQA, HotpotQA)
    - 3.1.2 Multi-Source Construction and Weighted Batch Sampling
    - 3.1.3 Dataset Statistics
- 3.2 Evaluation Data
    - 3.2.1 NanoBEIR: 13 Subsets
    - 3.2.2 NASA SDE IR and NASA SMD IR

---

## Chapter 4. Methodology

- 4.1 Base Models
    - 4.1.1 BGE-base-en-v1.5
    - 4.1.2 RoBERTa-base
- 4.2 Training Infrastructure
    - 4.2.1 Multi-GPU DDP with torchrun
    - 4.2.2 Hyperparameters and Configuration
- 4.3 Baseline Fine-Tuning (e1)
- 4.4 Evaluation Framework
    - 4.4.1 Embedding Cache and Transform Pipeline
    - 4.4.2 Metrics: nDCG, MRR, Accuracy, Recall at k
    - 4.4.3 Statistical Significance Testing (Wilcoxon)
- 4.5 Post-Hoc Compression Methods
    - 4.5.1 Scalar Quantization: INT8 and INT4
    - 4.5.2 Binary Quantization
    - 4.5.3 Product Quantization
    - 4.5.4 TurboQuant
- 4.6 Training-Time Compression Methods
    - 4.6.1 Binary-Aware Training with STE (e2)
    - 4.6.2 Matryoshka Representation Learning (e3)
    - 4.6.3 MRL + BAT Joint Training (e4)
    - 4.6.4 Annealed Tanh Binarization (e5) ← _novel_
    - 4.6.5 Multi-Bit Annealed Sigmoid Quantization (e6) ← _novel_

---

## Chapter 5. Experiments and Results

- 5.1 Evaluation Setup and Pareto Analysis Methodology
- 5.2 Baseline Results
    - 5.2.1 BGE-base-en-v1.5: Pre-Trained vs. Fine-Tuned
    - 5.2.2 RoBERTa-base: Pre-Trained vs. Fine-Tuned
- 5.3 Post-Hoc Compression
    - 5.3.1 INT8 and INT4
    - 5.3.2 Binary Quantization
    - 5.3.3 Product Quantization: Effect of Subspaces M
    - 5.3.4 TurboQuant: 1–8 Bits per Dimension
- 5.4 Training-Time Methods
    - 5.4.1 BAT vs. Post-Hoc Binary
    - 5.4.2 MRL Across Truncation Dimensions
    - 5.4.3 MRL + BAT Joint Compression
    - 5.4.4 Annealed Tanh vs. STE
    - 5.4.5 Multi-Bit Annealed Sigmoid (e6)
- 5.5 Cross-Method Pareto Analysis
    - 5.5.1 BGE-base-en-v1.5 Pareto Frontier
    - 5.5.2 RoBERTa-base Pareto Frontier
    - 5.5.3 Training-Time vs. Post-Hoc at Equivalent Bit Budgets
- 5.6 Domain-Specific Results
    - 5.6.1 NASA SDE IR
    - 5.6.2 NASA SMD IR
- 5.7 Discussion
    - 5.7.1 Recommended Method by Bit Budget
    - 5.7.2 Why BGE and RoBERTa Show Different Patterns
    - 5.7.3 Binary Constraints as Regularization

---

## Chapter 6. Conclusion and Future Work

- 6.1 Summary of Findings
- 6.2 Practical Guidance
- 6.3 Limitations
- 6.4 Future Work
    - 6.4.1 Unified MRL + Multi-Bit Framework
    - 6.4.2 Additional Base Models
    - 6.4.3 Latency Benchmarking
    - 6.4.4 Full BEIR Evaluation


---

# breakdown of what the content in each section should typically include:

1.1 Motivation
This section serves as the "hook" for your thesis. It should explain why your research area matters in the real world.

The Broad Context: Start with the big picture (e.g., the massive growth of large language models, recommendation engines, or search retrieval systems).

The Bottleneck: Introduce the specific pain point. In your case, explain how embeddings require massive amounts of memory, storage, and computational power, leading to high latency and infrastructure costs.

The "Why Now?": Explain why solving this problem is critical right now (e.g., deploying models on edge devices, reducing cloud costs, making AI more sustainable).

1.2 Embedding Compression Landscape
This section provides a high-level overview of the current solutions, setting the stage before you introduce your specific problem. (Note: This is not your full Literature Review, which usually gets its own chapter, but rather a primer).

Key Concepts: Briefly define what embedding compression is.

Current Paradigms: Categorize the existing approaches broadly (e.g., Quantization, Pruning, Knowledge Distillation, Dimensionality Reduction/PCA, or Hashing).

High-Level Trade-offs: Briefly mention the general pros and cons of these categories (e.g., "Method A is fast but loses accuracy; Method B retains accuracy but is slow to train").

1.3 Problem Statement
This is one of the most critical sections. It transitions from the general landscape to the exact gap your thesis addresses.

Identify the Gap: Point out a specific flaw, limitation, or unexplored area in the landscape you just described. For example: "While current quantization methods reduce size, they severely degrade retrieval accuracy for rare tokens."

State the Goal: Clearly state what your thesis intends to do about this gap. Keep it concise, formal, and tightly scoped.

1.4 Research Questions
Here, you break your Problem Statement down into specific, answerable, and measurable questions that your thesis will investigate.

Formulate 2 to 4 distinct questions (usually labeled RQ1, RQ2, etc.).

Example: "RQ1: How does applying non-uniform quantization to user-item embeddings affect recommendation accuracy compared to baseline models?"

Example: "RQ2: To what extent can compression ratio be increased before inference latency becomes a bottleneck?"

1.5 Contributions
This is where you explicitly state what you achieved or produced. It is essentially a summary of your results and novel additions to the field.

Use a bulleted list for clarity.

List tangible outputs: Include novel algorithms you developed, new theoretical proofs, comprehensive benchmark datasets you created, or open-source libraries you published.

State the impact: Briefly note the result of your contribution (e.g., "We introduce [Algorithm Name], which achieves a 4x reduction in memory footprint with less than a 1% drop in accuracy.").

1.6 Thesis Organization
This is a standard academic roadmap for the reader. It briefly outlines what to expect in the remaining chapters.

Keep it simple and direct.

Example: "The remainder of this thesis is organized as follows: Chapter 2 reviews the related literature on... Chapter 3 details the proposed methodology for... Chapter 4 presents the experimental setup and results... Chapter 5 concludes the thesis and discusses future work."