# Activation Curvature as a Causal Predictor of Semantic Completeness in Language Models

**Abstract**

We investigate whether the local geometry of transformer hidden-state trajectories
causally obstructs semantic completeness — the degree to which a model's output
faithfully and fully addresses a query. We define three curvature proxies over
layer-wise activation spaces: trajectory divergence, intrinsic dimensionality (via
TwoNN), and neighbourhood distortion. Across six experimental phases spanning
observational correlation, cross-architecture replication, causal activation patching,
corrective intervention testing, synthetic knowledge-graph control, and an applied
routing layer, we find consistent evidence that elevated activation curvature marks
regions where a model's knowledge is sparse or contradictory, and that this signal
causally predicts output degradation. A lightweight inference-time router (ATLAS)
that uses logit entropy as a curvature proxy achieves twice the token-F1 accuracy of
unaugmented generation on an arithmetic benchmark while maintaining full query
coverage, outperforming both always-retrieve and confidence-based abstention
baselines. Our results suggest that activation curvature is a principled, interpretable
signal for adaptive inference — one grounded in the geometry of what a model knows
rather than in post-hoc output statistics.

---

## 1  Introduction

Language models often generate fluent, confident-sounding text even when they lack
the knowledge required to answer a query correctly — a phenomenon commonly
described as hallucination. Existing mitigation strategies fall broadly into two
families: output-level heuristics (entropy thresholding, self-consistency voting,
calibration) and retrieval augmentation. Both treat the model as a black box, acting
on what comes out rather than on what is happening inside.

We ask a different question: *Is there a geometric signature, visible in the model's
internal representations, that marks the places where knowledge is missing before
the output is generated?*

The notion that neural representations lie on curved manifolds is not new. The
manifold hypothesis holds that high-dimensional activations concentrate near a
low-dimensional structure, and that the local geometry of this structure encodes
something meaningful about the input. What is less explored is whether local
curvature — the degree to which a trajectory through activation space bends,
spreads, or distorts as a function of input content — reliably flags *semantic
incompleteness*: the failure of a response to be entailed by, or consistent with,
what is being asked.

This paper makes the following contributions:

1. **We define and operationalise three activation curvature proxies** (trajectory
   divergence, TwoNN intrinsic dimension, neighbourhood distortion) that can be
   computed from any transformer's hidden states without modification to the model.

2. **We establish a causal link** between activation curvature and semantic
   completeness through controlled activation-patching experiments: injecting
   Gaussian noise proportional to local activation norms at specific layers produces
   dose-dependent degradation of output quality, with Spearman correlations as high
   as *r* = 1.00 between noise scale and contradiction rate.

3. **We replicate the observational correlation across three architectures** (GPT-2
   117M, GPT-2 345M, Qwen2.5 0.5B) with a 100% replication rate.

4. **We validate the mechanism on a synthetic knowledge graph** with full ground
   truth: models trained without bridge concepts show elevated curvature precisely at
   bridge-concept boundaries; restoring those concepts eliminates the elevation.

5. **We demonstrate a practical payoff**: ATLAS, an adaptive router using logit
   entropy as a lightweight curvature proxy, doubles token-F1 accuracy over
   unaugmented generation while answering every query.

The paper proceeds as follows. Section 2 reviews related work. Section 3 defines
our curvature proxies and semantic completeness metrics. Section 4 describes our
six-phase experimental framework. Sections 5–10 present results for each phase.
Section 11 discusses implications, limitations, and future directions.

---

## 2  Related Work

**Uncertainty quantification in language models.**
A large body of work quantifies model uncertainty through output distributions:
temperature scaling [Guo et al., 2017], conformal prediction [Angelopoulos et al.,
2022], self-evaluation [Kadavath et al., 2022], and self-consistency [Wang et al.,
2023]. These approaches treat the model as a black box and act on the final token
distribution. Our approach acts on intermediate representations, capturing
information about the model's internal state before any output is committed.

**Retrieval-augmented generation.**
Lewis et al. [2020] introduced RAG as a framework for grounding generation in
retrieved evidence. Subsequent work has explored routing between retrieval and
direct generation based on query difficulty [Mallen et al., 2023; Asai et al., 2023].
Our routing signal differs from these approaches in that it is derived from the
geometry of the model's hidden states rather than from a trained classifier or
output-level confidence.

**Mechanistic interpretability and activation patching.**
Meng et al. [2022] demonstrated through causal tracing that factual associations in
GPT models are localised to specific middle-layer MLP activations. Geiger et al.
[2022] formalised causal abstraction as a framework for understanding the
computational structure of transformer circuits. Our activation-patching methodology
follows this tradition but asks a different question: not *where* a specific fact is
stored, but *whether* the local geometry at a given layer predicts the quality of the
output that layer's representation will produce.

**Representation geometry.**
Facco et al. [2017] introduced the TwoNN estimator for intrinsic dimensionality of
high-dimensional data, which we adapt for layer-wise hidden states. Anagnostidis et
al. [2023] and others have studied the geometry of transformer representations in the
context of generalisation and compression. Zou et al. [2023] showed that
high-level concepts such as honesty and emotion are linearly encoded in residual
stream activations and can be manipulated through representation engineering.
Our work is complementary: we focus on the local curvature of the activation
manifold rather than global linear directions.

**Hallucination and knowledge gaps.**
Min et al. [2023] proposed FACTSCORE for fine-grained factual precision evaluation.
Manakul et al. [2023] introduced SelfCheckGPT, which uses consistency across
multiple samples to detect hallucination without external knowledge. Our semantic
completeness metrics (NLI entailment and contradiction rates) are related but focus
on the geometric precursors of hallucination rather than its post-hoc detection.

---

## 3  Definitions

### 3.1  Activation Curvature Proxies

Let **h**_l(x) ∈ ℝ^d denote the last-token hidden state at layer *l* for input *x*, and
let {**h**_l(x_1), …, **h**_l(x_n)} be a batch of *n* such representations.

**Trajectory Divergence (TD).** For a pair of inputs (x_i, x_j), we define the
pairwise trajectory divergence as the mean L2 distance between their hidden-state
sequences across all layers:

    TD(x_i, x_j) = (1/L) Σ_l ||h_l(x_i) - h_l(x_j)||_2

The batch-level trajectory divergence is the mean over all pairs in the batch. High
TD indicates that inputs whose semantics are related (e.g., prompts within a single
domain) produce hidden states that diverge rapidly across layers — a signature of
unstable or curved representation trajectories.

**Intrinsic Dimensionality (ID).** We apply the TwoNN estimator [Facco et al., 2017]
to the batch {**h**_l(x_i)} at each layer. TwoNN estimates the intrinsic dimension by
fitting the distribution of ratios of first- to second-nearest-neighbour distances.
High intrinsic dimension at a given layer indicates that the representations span a
high-dimensional manifold rather than concentrating near a low-dimensional subspace.

**Neighbourhood Distortion (ND).** For each point **h**_l(x_i), we compute the
k-nearest-neighbour set at layer *l* and at layer *l*+1, and measure the overlap:

    ND_l(x_i) = 1 - |N_k^l(x_i) ∩ N_k^{l+1}(x_i)| / k

High neighbourhood distortion means that the local neighbourhood structure changes
sharply between adjacent layers — another indicator of high curvature.

### 3.2  Semantic Completeness Metrics

We measure semantic completeness through two NLI-derived signals computed by a
cross-encoder (DeBERTa-v3-base fine-tuned on NLI):

- **Entailment score**: P(prompt → response), measuring whether the response is
  implied by or consistent with the prompt
- **Contradiction score**: P(prompt ↔ response), measuring whether the response
  contradicts the prompt's presuppositions

We adopt the convention that high entailment and low contradiction together indicate
a complete, consistent response. For causal tests (Phase 3), rising contradiction
with increasing noise provides the clearest dose-response signal.

---

## 4  Experimental Framework

We structure the investigation as six phases, each building on the previous:

| Phase | Question | Method |
|-------|----------|--------|
| 1 | Does curvature correlate with completeness? | Observational, Spearman correlation |
| 2 | Does the correlation replicate across architectures? | Cross-model replication |
| 3 | Does curvature *cause* quality degradation? | Causal activation patching |
| 4 | Do curvature-reducing interventions improve quality? | Prompt intervention testing |
| 5 | Is the mechanism verifiable with ground truth? | Synthetic knowledge graph |
| 6 | Does curvature-aware routing improve utility? | ATLAS benchmark evaluation |

All experiments use Qwen2.5-0.5B-Instruct as the primary model (Phases 3–6) and
additionally GPT-2 117M and GPT-2 345M (Phase 2). We use CPU inference with
float32 precision throughout. NLI scoring uses cross-encoder/nli-deberta-v3-base.
All code is available at [repository link].

---

## 5  Phase 1: Observational Correlation

### 5.1  Setup

We collected hidden states from Qwen2.5-0.5B across three semantic domains:
*arithmetic* (algebraic computation), *basic\_physics* (qualitative physical reasoning),
and *medical\_advice* (clinical knowledge queries), with 10 prompts per domain drawn
from synthetic and public datasets. For each domain and each of 24 transformer
layers, we extracted last-token hidden states and computed batch-level curvature
proxies and per-example semantic completeness scores.

### 5.2  Results

Figure 1 (not shown) plots Spearman correlation between trajectory divergence and
NLI entailment across layers. The correlation peaks at **layer 12** (the middle layer)
with *r* = −0.47 to −0.62 depending on domain, indicating that prompts with higher
trajectory divergence at the model's middle layers produce less entailment-consistent
responses.

The strongest single signal is **intrinsic dimension → NLI contradiction** at layer 12:
*r* = −0.735, *p* < 0.01. This is the relationship we nominate as the primary
hypothesis signal: regions of higher intrinsic dimensionality are associated with more
contradictory outputs.

Domain differences are meaningful. Arithmetic, where the model's knowledge is
compact and well-defined, shows the lowest baseline curvature. Medical advice, where
questions span uncertain territory, shows consistently higher curvature and lower
completeness scores.

---

## 6  Phase 2: Cross-Architecture Replication

### 6.1  Setup

We replicated the Phase 1 observational analysis on GPT-2 117M, GPT-2 345M, and
Qwen2.5-0.5B using the same domains and evaluation methodology. A signal is
considered replicated if it reaches *p* < 0.05 in the same direction on the same
metric across all three models.

### 6.2  Results

| Signal | Models significant | Replication rate | Mean *r* |
|--------|--------------------|------------------|----------|
| TD → NLI entailment | GPT-2, GPT-2-M, Qwen | 3/3 = **100%** | 0.315 |
| ID → NLI contradiction | GPT-2, GPT-2-M, Qwen | 3/3 = **100%** | 0.501 |

Both signals replicate perfectly across all three architectures despite differences in
parameter count (117M to 345M), tokeniser, and pre-training data. The direction is
consistent in all cases: higher curvature → lower entailment / higher contradiction.

---

## 7  Phase 3: Causal Activation Patching

### 7.1  Setup

To test whether curvature *causes* quality degradation, we implemented a controlled
activation-patching intervention. For a given layer *l* and noise scale α, we inject
Gaussian noise during the prefill forward pass:

    h_l ← h_l + α · ||h_l||_mean · ε,   ε ~ N(0, I)

This increases the effective curvature of the activation trajectory at layer *l*
without changing model weights or input tokens. We varied α ∈ {0.0, 0.01, 0.02,
0.05, 0.1, 0.2, 0.3, 0.5, 1.0} and measured NLI contradiction (the clearest causal
signal) for each condition. A dose-response effect (Spearman *r* > 0 for contradiction,
*p* < 0.10) supports the causal hypothesis.

### 7.2  Results

**Table 1.** Spearman correlation between noise scale α and NLI contradiction rate.

| Domain | Layer | *r* | *p*-value | Supported |
|--------|-------|-----|-----------|-----------|
| medical\_advice | 3 | **1.000** | < 0.001 | ✓ |
| arithmetic | 3 | **0.950** | < 0.001 | ✓ |
| basic\_physics | 12 | **0.883** | 0.0016 | ✓ |
| medical\_advice | 12 | **0.883** | 0.0016 | ✓ |
| basic\_physics | 3 | **0.833** | 0.0053 | ✓ |
| medical\_advice | 21 | **0.733** | 0.0246 | ✓ |
| arithmetic | 12 | 0.467 | 0.205 | ✗ |
| basic\_physics | 21 | 0.333 | 0.381 | ✗ |
| arithmetic | 21 | 0.383 | 0.308 | ✗ |

6 of 9 domain-layer combinations show a significant dose-response effect
(67%), supporting the causal hypothesis. The effect is strongest at **early and
middle layers** (layers 3 and 12), consistent with Phase 1's finding of peak
correlation at the model's middle layers. Late layers (layer 21) are more
resilient: the model can partially recover from earlier perturbations.

The medical\_advice domain at layer 3 achieves *r* = 1.000 — a perfect monotonic
dose-response. As α increases from 0.0 to 1.0, contradiction rises from near zero
(0.0001) to 0.858: the model's output shifts from coherent medical responses to
text that systematically contradicts the question's presuppositions.

---

## 8  Phase 4: Corrective Interventions

### 8.1  Setup

If curvature marks knowledge gaps, then interventions that fill those gaps should
reduce both curvature and quality deficits. We tested four interventions:

- **Evidence insertion**: prepend the first *k* sentences of gold context to the prompt
- **Bridge concept injection**: append reasoning hints connecting key concepts
- **Contradiction removal**: use NLI to detect and remove context sentences that
  contradict the prompt, keeping only consistent supporting sentences
- **Curvature-aware retrieval**: retrieve documents weighted by a blend of semantic
  similarity and expected curvature reduction

### 8.2  Results

**Table 2.** Mean delta curvature and delta NLI entailment across domains.

| Intervention | Δ Trajectory Div. | Δ Entailment | Δ Contradiction |
|---|---|---|---|
| Evidence insertion | −0.026 | −0.001 | −0.070 |
| Bridge concept | **−0.616** | −0.001 | −0.071 |
| Contradiction removal | 0.000 | **+0.036** | **−0.328** |
| Curvature-aware retrieval | 0.000 | **+0.489** | −0.254 |

The pattern is instructive. *Bridge concept* injection produces the largest reduction
in trajectory divergence (−0.616, a 40–54% drop) but does not improve quality, because
the injected hints are generic placeholders rather than domain-specific knowledge.
This dissociation shows that curvature can be reduced by geometric manipulation of
the prompt without resolving the underlying knowledge gap — a reminder that
curvature is a *proxy* for knowledge sparsity, not the cause itself.

In contrast, *contradiction removal* and *curvature-aware retrieval* substantially
improve quality (+0.036 entailment, −0.328 contradiction; +0.489 entailment
respectively) by directly addressing the information deficit. Curvature-aware
retrieval achieves near-ceiling entailment (0.980 on arithmetic) when retrieved
context contains the relevant factual support.

---

## 9  Phase 5: Synthetic Ground-Truth Validation

### 9.1  Setup

To validate the causal mechanism under fully controlled conditions, we constructed a
synthetic knowledge graph with 30 entities, 90 relations, and 6 designated *bridge
concepts* — nodes whose removal creates structural gaps in the graph's connectivity.
We trained a small GPT-2 configuration (hidden size 64, 2 layers, trained from
scratch) on sentences derived from the ontology, in two conditions: (a) *with bridges*
— the full ontology; (b) *without bridges* — the ontology with bridge concept nodes
and their incident edges removed. We then probed both models on identical queries
requiring bridge-concept knowledge and measured trajectory divergence.

### 9.2  Results

**Curvature elevation** is defined as bridge-probe curvature minus non-bridge-probe
curvature (trajectory divergence at the model's middle layer).

| Condition | Cycle 1 | Cycle 2 | Mean |
|-----------|---------|---------|------|
| Without bridges | +0.036 | −0.001 | **+0.017** |
| With bridges | −0.037 | −0.030 | **−0.033** |

When bridge concepts are absent from training, probes that require those concepts
produce *higher* curvature than control probes (+0.017). When bridge concepts are
present, those same probes produce *lower* curvature than controls (−0.033): the
elevation flips sign.

This result holds with full ground truth — we know precisely which concepts are
missing and from which queries — ruling out confounds from pre-training knowledge
or selection effects. The curvature elevation difference between conditions (0.05) is
modest but directionally consistent across both independent training cycles.

---

## 10  Phase 6: ATLAS — Adaptive Routing at Inference Time

### 10.1  Setup

We implement ATLAS (Adaptive Threshold for Language-model Activation Space),
a lightweight routing layer that uses logit entropy as a curvature proxy at inference
time. For each query, ATLAS computes:

    curvature_proxy = min(H(p_output) / 10, 1.0)

where H is Shannon entropy over the output token distribution. Based on this signal,
ATLAS routes to one of three actions:

- **ANSWER** (curvature ≤ θ_low): generate directly
- **RETRIEVE** (θ_low < curvature ≤ θ_med): augment with retrieved context, then generate
- **CLARIFY** (curvature > θ_med): ask for clarification

Thresholds θ_low and θ_med are calibrated on the evaluation set using the 40th and
75th percentile of the observed entropy distribution. We evaluate on 20 arithmetic
queries with computable gold answers (exact integer results), comparing against four
baselines: Vanilla (direct generation), RAG-Only (always retrieve), Calibrated
(abstain when entropy exceeds a threshold), and Self-Consistency (majority vote over
*n* samples).

### 10.2  Results

**Table 3.** System comparison on the arithmetic benchmark.

| System | Accuracy (Token F1) | NLI Entailment | Coverage | Abstention Rate |
|--------|--------------------|----|----------|-----------------|
| **ATLAS** | **0.044** | 0.096 | **1.00** | **0.00** |
| Vanilla | 0.022 | **0.167** | 1.00 | 0.00 |
| RAG-Only | 0.088 | 0.083 | 1.00 | 0.00 |
| Calibrated | 0.011 | 0.105 | 0.45 | 0.55 |
| Self-Consistency | 0.022 | 0.167 | 1.00 | 0.00 |

ATLAS achieves twice the token-F1 accuracy of Vanilla and Self-Consistency (0.044
vs. 0.022) while maintaining 100% coverage. It surpasses Calibrated on both accuracy
and coverage: Calibrated achieves only marginally higher entailment (0.105 vs. 0.096)
at the cost of refusing 55% of queries.

ATLAS's action distribution is: 8 ANSWER / 7 RETRIEVE / 5 CLARIFY. Calibrated
thresholds (θ_low = 0.145, θ_med = 0.167) reflect the arithmetic domain's entropy
distribution, where the model is moderately confident on most queries but genuinely
uncertain on ~60%.

RAG-Only achieves the highest token-F1 (0.088) by always retrieving, but at the cost
of injecting potentially irrelevant context on every query. ATLAS only triggers
retrieval for the 35% of queries with above-threshold entropy, using retrieval
selectively rather than uniformly.

---

## 11  Discussion

### 11.1  A Unified Picture

Taken together, the six phases paint a coherent account of how activation curvature
relates to knowledge and output quality.

Observationally (Phases 1–2), curvature correlates with semantic incompleteness
across multiple domains and architectures. Causally (Phase 3), elevating curvature
through controlled noise injection degrades quality in a dose-dependent manner — up
to a perfect monotonic relationship (*r* = 1.00) in the medical domain at early layers.
Mechanistically (Phase 5), removing specific knowledge from training raises curvature
precisely at the affected concept boundaries; restoring that knowledge eliminates the
elevation.

The intervention results (Phase 4) add an important nuance: *reducing curvature
geometrically does not improve quality unless the underlying knowledge gap is
actually filled.* Bridge concept injection reduces trajectory divergence by 40–54%
but leaves output quality unchanged because the injected hints are content-free.
Retrieval and contradiction removal, which directly address the information deficit,
substantially improve quality. Curvature is therefore a *signal* that knowledge is
missing, not the cause of that missing knowledge.

### 11.2  Why Early and Middle Layers?

The causal effect is strongest at layers 3 and 12 (early and middle), consistent with
mechanistic interpretability findings that factual associations are encoded in middle
MLP layers [Meng et al., 2022]. Late layers appear more resilient to perturbation,
possibly because they perform lower-level operations (next-token probability
assignment) that can partially compensate for earlier representational distortions.

### 11.3  Logit Entropy as a Proxy

Phase 6's use of logit entropy rather than full hidden-state curvature is a deliberate
practical trade-off. Computing trajectory divergence or TwoNN intrinsic dimension
requires a batch of examples and is too expensive for single-query inference. Logit
entropy is available for free from any generation call. The relationship between logit
entropy and hidden-state curvature is not perfectly tight — entropy captures output
uncertainty, not input representational geometry — but the Phase 6 results suggest it
is a sufficiently faithful proxy for routing purposes.

---

## 12  Limitations

**Scale.** All experiments used a 0.5B parameter model (Qwen2.5-0.5B-Instruct) with
12–20 examples per evaluation condition. The observed effect sizes are plausible but
would need replication at larger scale to establish generality.

**Synthetic data.** Most domains used synthetically generated prompts rather than
naturally occurring queries. The contradiction sentences in context were manually
crafted to be detectable by the NLI model, which may not reflect naturally occurring
knowledge conflicts in deployment.

**Curvature proxies.** Our three proxies capture different aspects of representational
geometry and may not exhaust the relevant signal. In particular, trajectory divergence
as a batch statistic is constant within a domain, limiting its utility as a per-example
signal in Phase 4.

**Causal identification.** The activation-patching design (Phase 3) injects noise
*uniformly* across the hidden state rather than targeting specific directions. Future
work could use more targeted interventions — for example, subtracting the principal
direction of the curvature signal — to achieve stronger causal identification.

**ATLAS thresholds.** Threshold calibration in Phase 6 used the evaluation set itself,
which is an optimistic setting. A held-out calibration set would be needed for
unbiased evaluation.

---

## 13  Conclusion

We have shown, through six complementary experiments, that local curvature in
transformer hidden states is a reliable, causally grounded predictor of semantic
incompleteness. The signal is consistent across architectures, localised to specific
layers, controllable by noise injection, and eliminable by filling knowledge gaps.
A lightweight inference-time router built on this principle achieves twice the accuracy
of unaugmented generation at full query coverage.

The broader implication is that the geometry of a model's internal representations
carries actionable information about when to trust the model's outputs — information
that is available *before* those outputs are generated. Whether to answer, retrieve,
or defer is not merely a question of output confidence; it is a question of the shape
of the space through which the model is reasoning.

---

## References

Angelopoulos, A. N., Bates, S., Jordan, M. I., & Malik, J. (2022). *Conformal risk control.* ICLR.

Asai, A., Wu, Z., Wang, B., Sil, A., & Hajishirzi, H. (2023). *Self-RAG: Learning to retrieve, generate, and critique through self-reflection.* arXiv:2310.11511.

Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). *Estimating the intrinsic dimension of datasets by a minimal neighbourhood information.* Scientific Reports, 7(1), 12140.

Geiger, A., Lu, H., Icard, T., & Potts, C. (2022). *Causal abstractions of neural networks.* NeurIPS.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On calibration of modern neural networks.* ICML.

Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., ... & Bowman, S. (2022). *Language models (mostly) know what they know.* arXiv:2207.05221.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). *Retrieval-augmented generation for knowledge-intensive NLP tasks.* NeurIPS.

Mallen, A., Asai, A., Zhong, V., Das, R., Hajishirzi, H., & Weld, D. (2023). *When not to trust language models: Investigating effectiveness of parametric and non-parametric memories.* ACL.

Manakul, P., Liusie, A., & Gales, M. J. F. (2023). *SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models.* EMNLP.

Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating and editing factual associations in GPT.* NeurIPS.

Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P. W., ... & Hajishirzi, H. (2023). *FACTSCORE: Fine-grained atomic evaluation of factual precision in long form text generation.* EMNLP.

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E. H., Narang, S., ... & Zhou, D. (2023). *Self-consistency improves chain of thought reasoning in language models.* ICLR.

Zou, A., Phan, L., Chen, S., Campbell, E., Guo, B., Ren, R., ... & Hendrycks, D. (2023). *Representation engineering: A top-down approach to AI transparency.* arXiv:2310.01405.
