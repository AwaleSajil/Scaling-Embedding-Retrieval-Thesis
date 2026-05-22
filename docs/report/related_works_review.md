# Critique of Related Works (`ch2_background.tex`)

Overall, the Related Works section is well-structured and clearly sets up the context for the thesis. It successfully navigates through post-hoc quantization, training-time binarization, MRL, and controlled evaluation. However, there are a few areas that can be improved to enhance clarity and impact.

## 1. Explicitly Enumerate the "Three Gaps"
**Current State:** On line 394, the text states, *"This thesis addresses all three gaps..."* However, the exact three gaps are scattered throughout the preceding paragraph and aren't explicitly numbered.
**Recommended Change:** Right before that sentence, explicitly list the three gaps to make the thesis contributions punchier. For example:
> "In summary, the existing literature leaves three critical gaps: (1) no prior study enables a direct comparison between post-hoc and training-time compression under controlled conditions; (2) prior studies evaluate differing single backbones, leaving cross-architecture generalization unstudied; and (3) stacked combinations of training-time methods, such as MRL jointly trained with binarization, remain unexplored. This thesis addresses all three gaps by..."

## 2. Improve Transition from Foundations to IR Applications
**Current State:** Paragraph 2 (lines 322-341) discusses the foundational neural network binarization papers (Bengio, Courbariaux, Cao, Shah, Yang, Schrödter). Paragraph 3 (line 343) abruptly begins with *"Both STE and annealed tanh have been adopted for dense retrieval."*
**Recommended Change:** Add a transitional sentence at the beginning of Paragraph 3 to bridge the gap between general neural network compression and Information Retrieval.
> "While the aforementioned gradient approximation strategies were initially developed for general neural network weights and activations, they naturally extend to the compression of dense retrieval representations. Both STE and annealed tanh have been adopted..."

## 3. Sharpen the Contrast with QAMA
**Current State:** Paragraph 4 discusses QAMA (Quantization-Aware Matryoshka Adaptation) and correctly notes that it evaluates on ModernBERT and MiniLM, leaving open the question of generalization.
**Recommended Change:** Given that QAMA is the closest prior work (combining MRL and quantization), you should also explicitly contrast your *methodological* approach. For example, mention that while QAMA relies on complex hybrid precision and learnable thresholds, your work focuses on whether simpler, uniform training-time binarization (STE/Tanh) can be effectively stacked with MRL.

## 4. Condense the Deep-Dive on Multi-bit Quantization
**Current State:** Paragraph 2 spends several lines discussing Yang et al. (shifted sigmoid functions) and Schrödter et al. (sigmoid staircase for input features).
**Recommended Change:** Since your thesis focuses on INT8, Binary, TurboQuant, and PQ (as per Table 2.1), the deep dive into sigmoid staircases for multi-bit quantization slightly distracts from your core narrative (STE and Tanh). Consider condensing those two sentences into a single, brief mention of multi-bit extensions, keeping the focus tightly on the binary surrogates that you actually use in your experiments.
