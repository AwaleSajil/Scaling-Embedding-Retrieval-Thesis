# Learnable threshold for multibit quantization traning (Literature review)

Quantization Networks (CVPR 2019)
arxiv.org/abs/1911.09464
This is the canonical paper for the sum-of-sigmoids approach. They represent a k-level quantizer as a weighted sum of k-1 sigmoid functions with learnable biases (= thresholds), then anneal a temperature upward so each sigmoid sharpens into a step function. Exactly what you'd be implementing for multi-bit. Applied to weights and activations for 2/3/4-bit networks.

Trainable Bitwise Soft Quantization (2025)
arxiv.org/abs/2603.05172
Very recent, most directly analogous to your setup — applies the sigmoid staircase to compress neural network input features end-to-end. Each bit threshold is a learned sigmoid, temperature-annealed to a hard step at inference. 5–16x compression with modest accuracy loss.

QAMA — Quantization Aware Matryoshka Adaptation (CIKM 2025)
dl.acm.org/doi/10.1145/3746252.3761077
The closest to your exact thesis: combines MRL (which you already have) with multi-level quantization-aware training for retrieval embeddings. At 192 dims + 2-bit, retains 96% of full-precision nDCG@10.

Adaptive thresholds for hashing (pre-deep learning, but foundational)
Double-Bit Quantization for Hashing — DBQ (AAAI 2012)
ojs.aaai.org/index.php/AAAI/article/view/8208
The original "2-bit hashing" paper. Key insight: the 1-bit threshold at 0 sits at the highest-density region of the distribution, causing nearby points to flip bits unpredictably. DBQ places two thresholds symmetrically around 0 to avoid this. Thresholds are set from data statistics, not gradient-based, but it's the direct precursor to what you want.

Unsupervised Online Hashing with Multi-Bit Quantization (ACCV 2022)
openaccess.thecvf.com/content/ACCV2022/papers/Weng...
Adaptive per-dimension threshold placement for retrieval — thresholds are placed to match the empirical distribution of each projection. Directly addresses "where to put the thresholds when the range is unknown."

Learned step size / threshold training (weight quantization, but relevant techniques)
LSQ — Learned Step Size Quantization (ICLR 2020)
arxiv.org/abs/1902.08153
Learns a single scale per layer with a straight-through gradient. The dominant baseline for learned uniform quantization. Uniform spacing (not arbitrary thresholds), but the technique for differentiating through quantization is directly applicable.

QIL — Quantization Interval Learning (CVPR 2019)
arxiv.org/abs/1808.05779
Learns the actual interval boundaries (thresholds) by backpropagating the task loss — the direct ancestor of the "anchor + softplus gaps" parameterization.

Practitioner blog posts
HuggingFace: Binary and Scalar Embedding Quantization (2024)
huggingface.co/blog/embedding-quantization
The standard reference for 1-bit binary (sign at 0) and int8 scalar quantization for sentence transformers. This is the baseline you're improving on.

Mixedbread: Binary MRL (2024)
mixedbread.com/blog/binary-mrl
Joint training of Matryoshka + binary quantization (same combination as your e4/e5 experiments). Shows training with binarization in mind — not post-hoc — is critical.

The gap your thesis fills
Almost no paper combines: (1) annealed soft quantization + (2) multi-bit learned thresholds + (3) sentence transformer finetuning for retrieval. QAMA is the closest (2025), but uses a penalty loss rather than a soft differentiable staircase. Quantization Networks has the architecture, but applied to vision model weights, not retrieval embeddings. That gap is what makes the experiment worth running.

---

# Implementation Plan: Multi-bit Annealed Tanh (e6)

## Core idea

Generalise `AnnealedTanhBinarizationLayer` (1-bit, one threshold at 0) to B bits using a differentiable staircase: B-1 shifted tanh functions, each approximating one decision boundary of a uniform B-level quantizer. As β grows, each tanh sharpens into a step, and the whole staircase converges to a hard round-to-nearest quantizer at inference.

For B bits, M = 2^B levels, step size Δ = 2/(M−1), fixed thresholds t_k = −1 + (k + 0.5)·Δ for k = 0 … M−2:

```
q_soft(x) = Δ · Σ_{k=0}^{M-2} σ(β · (x − t_k))  −  (M−1)/2 · Δ
```

where σ is sigmoid (equivalent to a scaled tanh). At inference: `round_to_nearest_level(x)`.

B=1 recovers the existing e5 formula (`tanh(β·x)`) up to a constant scale factor.

**Key design issue — value range**: L2-normalised BGE embeddings have σ ≈ 0.036 per dimension (observed in the embedding distribution notebook). Fixed uniform thresholds spanning [-1, 1] would collapse all values to a single bin. Two options:

- **Learnable thresholds** (preferred): initialise at ±σ/2, ±3σ/2, … and learn them via backprop through the staircase. `nn.Parameter` vector of length M−1, kept sorted via `softplus` gaps (as in QIL, CVPR 2019).
- **Scaled fixed thresholds**: space thresholds at multiples of σ, estimated from a warm-up batch. Simpler, but less expressive.

Start with learnable thresholds.

---

## Learnable threshold design

### Parameterisation: anchor + softplus(log_gaps)

We need M−1 threshold parameters t_0 < t_1 < … < t_{M−2} that remain strictly sorted throughout training. Storing them directly as unconstrained parameters doesn't guarantee ordering. Instead, use:

```
t_0     = anchor                                (nn.Parameter, scalar)
t_k     = anchor + Σ_{j=0}^{k-1} softplus(log_gaps[j])   for k ≥ 1
gaps[j] = softplus(log_gaps[j]) = log(1 + exp(log_gaps[j]))   > 0  always
```

`anchor` and `log_gaps` (shape: M−2) are the actual `nn.Parameter`s. Because each gap is strictly positive (softplus has range (0, ∞)), the resulting thresholds are strictly increasing by construction. The raw parameters `log_gaps` are unconstrained reals, so gradient descent works normally.

Why `softplus` and not `exp`? `exp` overflows for large inputs and has exploding gradients. `softplus` is numerically stable and its gradient is σ(log_gaps[j]) ∈ (0, 1), so the gradient of any threshold t_k with respect to log_gaps[j] (j < k) is always well-conditioned.

Concrete example for 2-bit (M=4, 3 thresholds, 2 gaps):
```
anchor       →  t_0
log_gaps[0]  →  gap_0 = softplus(log_gaps[0])  →  t_1 = t_0 + gap_0
log_gaps[1]  →  gap_1 = softplus(log_gaps[1])  →  t_2 = t_1 + gap_1
```
Total parameters for the threshold layer: 1 + (M−2) = M−1 scalars. Very cheap.

### Initialisation

The goal is to cover the actual value range of L2-normalised BGE embeddings (σ ≈ 0.036, mean ≈ 0) with one threshold per quantile boundary. Place them uniformly across [−init_scale, +init_scale] where `init_scale = σ`:

| bits | M | thresholds (linspace(−0.036, 0.036, M−1)) |
|------|---|-------------------------------------------|
| 2    | 4 | {−0.036, 0, +0.036} |
| 3    | 8 | {−0.036, −0.0257, −0.0129, 0, +0.0129, +0.0257, +0.036} |
| 4    | 16 | 15 values uniformly in [−0.036, +0.036] |

From the linspace, extract `anchor = thresholds[0]` and `raw_gaps = diff(thresholds)` (all equal at init). Since `softplus(log_gaps) = raw_gaps`, initialise `log_gaps = softplus_inverse(raw_gaps) = log(exp(raw_gaps) − 1)`.

This ensures at step 0 the staircase thresholds are exactly at the right scale for the data, and each quantisation bin captures roughly the same fraction of the embedding value distribution.

### Output levels vs thresholds

The thresholds control *where* the decision boundaries are. The output *values* are determined by the staircase formula. As β→∞, the output for a value x in bin k (between t_{k−1} and t_k) converges to:

```
q_hard(x in bin k) = (k + 1 − (M−1)/2) · Δ      where Δ = 2/(M−1)
```

For 2-bit: {−1, −1/3, +1/3, +1}.  For 3-bit: {−1, −5/7, −3/7, −1/7, +1/7, +3/7, +5/7, +1}.

These output values are fixed regardless of where the thresholds sit — the thresholds only decide which bin each input falls into. The output is always in [−1, +1], so cosine similarity on the quantised vectors is well-defined.

Note: the quantised embedding vector is no longer unit-norm (input was L2-normalised, quantisation changes component magnitudes). This is fine because `cos_sim` in the loss normalises internally. The cosine distance between two quantised vectors is a meaningful similarity as long as the threshold positions are informative (i.e., they separate the embedding space in a direction-preserving way).

### Gradient dynamics and the peaking problem

During training the loss flows back through:

```
loss  →  q_soft  →  sigmoid(β · (x − t_k))  →  t_k  →  anchor, log_gaps
```

The gradient of `q_soft` w.r.t. threshold t_k is:

```
∂q_soft/∂t_k = −Δ · β · σ'(β · (x − t_k))
```

where σ'(z) = σ(z)(1−σ(z)) is the sigmoid derivative, peaked at z=0 with max value 0.25 and decaying rapidly away from 0.

**The peaking problem**: as β grows, σ'(β·(x−t_k)) becomes very narrow (width ∝ 1/β). A threshold t_k only receives a meaningful gradient from inputs x satisfying |x − t_k| ≲ 2/β. At β=28 (end of training with γ=0.1), the effective gradient window is ≈ ±0.07 around each threshold. Since embedding values have σ=0.036, thresholds that drift outside [−0.1, +0.1] will receive almost no gradient and effectively freeze.

This creates a critical dependency on early-training dynamics: thresholds must reach good positions *before* β gets large.

### Warm-up strategy

Add a `warmup_steps` parameter (suggested: 500–1000 steps) during which β is frozen at 1.0. The staircase is fully soft, gradients are wide, and thresholds can move freely to data-driven positions. Annealing only starts after warmup completes:

```python
@property
def beta(self) -> float:
    anneal_steps = max(0, self._step - self.warmup_steps)
    return (self.gamma * anneal_steps + 1) ** 0.5
```

During warmup, β=1 regardless of γ. This is a small change to `AnnealedTanhBinarizationLayer`'s schedule but has a large effect on threshold stability.

Suggested value: `warmup_steps = 500` (≈6% of the 8k total steps, consistent with the 10% warmup_ratio already used in `trainer_config`).

### Failure modes to monitor via W&B

Log these at every `logging_steps` in `BetaAnnealCallback`:

| Metric | What to watch for |
|--------|-------------------|
| `multibit_atanh/thresholds` (all M−1 values) | Any two thresholds converging (gap < 1e-4) |
| `multibit_atanh/threshold_span` = t_{M−2} − t_0 | Span shrinking below σ/2 or growing beyond 5σ |
| `multibit_atanh/min_gap` | If min gap → 0: thresholds collapsing, increase init_scale or add gap regularisation |
| `multibit_atanh/beta` | Normal growth curve — flag if it plateaus unexpectedly |

If threshold collapse is observed, add an L2 regularisation term on the gap reciprocals to the loss: `λ · Σ_k 1/gap_k`, which penalises very small gaps. A small λ (e.g. 1e-4) is usually sufficient.

---

## Step 1 — New layer: `MultibitAnnealedTanhLayer` in `src/utils/utils.py`

Model after `AnnealedTanhBinarizationLayer`. Key differences:

```python
class MultibitAnnealedTanhLayer(nn.Module):
    def __init__(self, bits: int = 2, gamma: float = 0.1,
                 init_scale: float = 0.036, warmup_steps: int = 500):
        super().__init__()
        self.bits = bits
        self.gamma = gamma
        self.warmup_steps = warmup_steps
        self._step = 0
        M = 2 ** bits
        self.delta = 2.0 / (M - 1)

        # Thresholds parameterised as anchor + cumsum(softplus(log_gaps)).
        # Initialised uniformly across [-init_scale, +init_scale].
        init_t = torch.linspace(-init_scale, init_scale, M - 1)
        self.anchor = nn.Parameter(init_t[0].clone())
        if M > 2:
            raw_gaps = torch.diff(init_t)                    # all equal, all positive
            # softplus_inverse: log(exp(x) - 1)
            self.log_gaps = nn.Parameter(torch.log(torch.exp(raw_gaps) - 1))
        else:
            self.log_gaps = None                             # 1-bit: no gaps needed

    @property
    def thresholds(self) -> torch.Tensor:
        if self.log_gaps is None:
            return self.anchor.unsqueeze(0)
        gaps = torch.nn.functional.softplus(self.log_gaps)  # always > 0
        return self.anchor + torch.cat([
            torch.zeros(1, device=self.anchor.device),
            torch.cumsum(gaps, dim=0)
        ])

    @property
    def beta(self) -> float:
        anneal_steps = max(0, self._step - self.warmup_steps)
        return (self.gamma * anneal_steps + 1) ** 0.5       # frozen at 1.0 during warmup

    def anneal_step(self):
        self._step += 1

    def forward(self, features):
        x = features["sentence_embedding"]              # (batch, dim)
        t = self.thresholds                             # (M-1,)
        if self.training:
            logits = self.beta * (x.unsqueeze(-1) - t)  # (batch, dim, M-1)
            q = torch.sigmoid(logits).sum(dim=-1) * self.delta \
                - (2 ** self.bits - 1) / 2 * self.delta
            features["sentence_embedding"] = q
        else:
            M = 2 ** self.bits
            levels = torch.linspace(-1.0, 1.0, M, device=x.device)
            dists = (x.unsqueeze(-1) - levels).abs()    # (batch, dim, M)
            features["sentence_embedding"] = levels[dists.argmin(dim=-1)]
        return features

    # save / load / get_config_dict — same pattern as AnnealedTanhBinarizationLayer.
    # Persist: bits, gamma, warmup_steps, _step, anchor.item(), log_gaps.tolist()
```

Add `anneal_step()` so the existing `BetaAnnealCallback` in `train.py` works unchanged (it searches for any `AnnealedTanhBinarizationLayer`; update the isinstance check to also match `MultibitAnnealedTanhLayer`).

---

## Step 2 — New experiment branch `e6` in `src/train.py`

In `initilize_model()`, add after the e5 branch:

```python
elif config["experiment"]["number"] == "e6":
    word_embedding_model = models.Transformer(config["input_model"]["name"])
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    quant_layer = MultibitAnnealedTanhLayer(
        bits=config["multibit_atanh_config"]["bits"],
        gamma=config["multibit_atanh_config"]["gamma"],
        init_scale=config["multibit_atanh_config"]["init_scale"],
    )
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model, quant_layer], ...
    )
```

Update the `BetaAnnealCallback` isinstance check to cover both layer types:

```python
anneal_layer = next(
    (m for m in model.modules()
     if isinstance(m, (AnnealedTanhBinarizationLayer, MultibitAnnealedTanhLayer))), None
)
```

Loss: plain `MultipleNegativesRankingLoss` with `cos_sim` — same as e5. The output is multi-level float (not binary), so Hamming does not apply. Cosine similarity is correct.

W&B logging: reuse `annealed_tanh/beta` key; optionally add `multibit_atanh/thresholds_min` and `thresholds_max` to track threshold drift.

---

## Step 3 — Config: `src/config.yaml`

Add below the existing `annealed_tanh_config` block:

```yaml
multibit_atanh_config:
  bits: 2          # start here; sweep 2, 3, 4
  gamma: 0.1       # same annealing rate as e5 baseline
  init_scale: 0.036  # matches observed embedding σ (see analysis/1_embedding_distribution.ipynb)
```

---

## Step 4 — Model registry: `eval_v2/config/models.py`

Add `atanh_bits: Optional[int] = None` to `ModelSpec`. Register one entry per bit width (and per MRL dimension if combining with e3):

```python
"finetuned-bge-base-en-v1.5-multibit-atanh-2bit": ModelSpec(
    path="<timestamp>/final_model",
    display_name="AnnTanh-2b",
    group="AnnealedTanh",
    color="<new color>",
    hatch="+-",
    marker="bowtie",
    similarity="cosine",    # NOT hamming — multi-level float output
    atanh_bits=2,
),
# repeat for 3bit, 4bit
```

No new transform in `eval_v2/wrappers/transforms.py` is needed. The `MultibitAnnealedTanhLayer` applies hard quantization inside the model's `.eval()` forward pass, so the embeddings stored in the cache are already quantized floats.

---

## Step 5 — Run experiment script: `experiments/e6_run_multibit_atanh.sh`

```bash
torchrun --nproc_per_node=4 ../src/train.py --expirement_number e6
```

---

## Suggested sweep

| Run | bits | gamma | notes |
|-----|------|-------|-------|
| e6-2b | 2 | 0.1 | direct comparison with e5 (1-bit) |
| e6-3b | 3 | 0.1 | 8 levels |
| e6-4b | 4 | 0.1 | 16 levels — likely diminishing returns |
| e6-2b-fast | 2 | 0.5 | faster annealing, check stability |

Compare all against `bge-base-en-v1.5` (base), `finetuned-bge-base-en-v1.5` (FT), and `finetuned-bge-base-en-v1.5-atanh-gamma0.1` (e5, 1-bit) on NanoBEIR nDCG@10.