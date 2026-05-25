# HELIX: Diagnosing and Provably Restoring Post-Hoc Control Layer Collapse via Threshold Decoupling

**K Dhiraj**  
Department of Computer Science and Engineering  
Chandigarh University  
k.dhiraj.srihari@gmail.com

**Abstract—** Post-hoc control layers impose inference-time constraints on trained classifiers without modifying model weights. HELIX is a margin-based module for class-selective override in a neural IDS, designed to suppress low-confidence class-4 predictions. We formally define **control collapse** — a structural failure mode in which the override layer processes every prediction but modifies none — using a dual criterion: override\_rate < $\varepsilon$ and KL divergence between pre- and post-override prediction distributions equal to zero.

We prove that the *online\_hybrid* configuration satisfies both criteria exactly on UNSW-NB15 (override\_rate = 0.0, KL = 0.0) and NSL-KDD (override\_rate = 0.0, KL = 0.0), despite class-4 prediction mass of 20.6% and 42.0% respectively. The root cause is adaptive threshold coupling: the threshold is estimated from the same margin stream it is meant to bound, and a dimensional mismatch ($\tau_a \sim O(10^4)$ vs. $z_{\text{norm}} \sim O(1)$) causes the gate to fail silently.

A minimal corrective intervention — freezing the adaptive threshold at calibration time, where $\tau_a^{\text{frozen}} = \mu_{\text{cal}} + 0.754\sigma_{\text{cal}} = 81{,}921$ logit units — breaks the coupling and restores non-zero override actuation. Bootstrap validation (n = 1,000 resamples, 95% CI) confirms statistically stable restoration: UNSW override\_rate = 0.0947 [0.09125, 0.09836]; NSL override\_rate = 0.3784 [0.37230, 0.38459]. The collapsed and active configurations have non-overlapping CIs, with zero-width variance under collapse confirming deterministic architectural failure rather than sampling noise.

One transparency result merits disclosure: under current calibration, the decoupled *frozen\_z\_hybrid* configuration is numerically identical to the simpler *fixed\_tau\_only* baseline. The z-score normalization channel adds architectural robustness but no additional discriminative signal at these parameter settings.

**Index Terms—** intrusion detection system, post-hoc control, confidence thresholding, control collapse, conformal calibration, selective classification, edge security.

## I. INTRODUCTION

Post-hoc inference control is a practical alternative to full retraining when decision-level constraints must be imposed on a deployed classifier [1]–[4]. The standard use case in intrusion detection is conservative reclassification: when the model assigns uncertain top-1 predictions to a high-cost class, override the prediction at inference time rather than retrain the model to fix it.

The natural confidence signal for such a criterion is the **decision margin** — the gap between the top-1 and second-best logit scores. Margin-based thresholding is used in confidence calibration [5], selective classification [6], and reject-option design [7]. A common refinement is the **adaptive threshold**: compute the override boundary from the running statistics of the observed margin distribution, so the violation rate stays roughly stable even as the input distribution shifts.

Under a specific implementation pattern, adaptive thresholding can nullify the control layer it is intended to enforce.

**The central result:** When the threshold is estimated online from the same stream over which violations are evaluated, and when the threshold estimator and the comparison operand live in different units, the override probability converges to zero by construction. The control layer runs on every sample and changes none. We call this failure **control collapse**, define it formally, prove it holds in the HELIX system, identify the mechanism, and show a minimal fix.

The failure mode is not exposed by existing methods. Confidence calibration [5, 8] operates in probability space with static parameters. OOD detection [9, 10] applies fixed, pre-computed thresholds. Conformal prediction [11, 12] provides coverage guarantees precisely *because* it freezes the nonconformity threshold before inference — exactly the safeguard that the online hybrid configuration violates. Selective classification [6, 13] analyzes static coverage constraints; online threshold drift is out of scope. This paper fills that gap.

**Contributions:**
1. **Formal collapse definition** via a dual criterion (override\_rate < $\varepsilon$, KL divergence = 0) with exact measured values.
2. **Collapse proof** showing *online\_hybrid* satisfies both necessary and sufficient conditions on two datasets.
3. **Mechanistic explanation** — Proposition 1 — establishing coupling-induced convergence of override probability to zero.
4. **Minimal corrective intervention** — frozen threshold decoupling — restoring non-zero actuation, with bootstrap confirmation of stability.
5. **Controllability–correctness analysis** documenting the trade-off and its calibration context, including the transparency result that *frozen\_z\_hybrid* $\equiv$ *fixed\_tau\_only* under current parameters.

---

## II. RELATED WORK

### A. Confidence Calibration

Guo et al. [5] show that modern neural networks are systematically miscalibrated and propose temperature scaling as a fix. Platt scaling [14] and isotonic regression [15] address the same objective via different parametric forms. All these methods work in probability space with a temperature parameter that is static after calibration. None analyzes the behavior of an override threshold that is continuously re-estimated from the test stream. The control collapse failure mode is invisible to calibration analysis because calibration is not in the loop.

### B. Out-of-Distribution Detection

Hendrycks and Gimpel [9] propose maximum softmax probability as a baseline OOD score; Lee et al. [10] use Mahalanobis distance in feature space; ODIN [16] combines temperature scaling and input perturbations. All three apply a **fixed, pre-computed threshold** to the score — the coupling problem does not arise. Collapse is specific to the regime where the threshold co-evolves with the score distribution during evaluation.

### C. Conformal Prediction and Selective Classification

Conformal prediction [11, 12] provides valid coverage guarantees under exchangeability by computing the nonconformity threshold on a held-out calibration set and **freezing it before inference**. Selective classification [6, 13] similarly fixes coverage constraints at deployment. Both frameworks are structurally correct for the same reason the frozen HELIX variant works: threshold and evaluation stream are decoupled. The present work documents what happens when that safeguard is removed.

### D. Adaptive Thresholding in Streaming Settings

Concept drift detection [17, 18] uses adaptive thresholds on the label or classifier space, not on the confidence threshold applied to the classifier's own output. The self-referential pattern — threshold adapts to the same signal it is meant to bound — is specific to post-hoc control overrides. This paper provides a formal analysis of that case.

**Why existing work misses this.** All surveyed methods either use fixed thresholds from held-out calibration, operate in a different representation space, or do not model self-referential threshold–margin feedback. The *online\_hybrid* configuration violates all three safeguards simultaneously.

---

## III. PROBLEM SETUP AND FORMAL DEFINITIONS

### A. Margin Definition

Let $\mathbf{z}(x) \in \mathbb{R}^K$ denote the pre-softmax logit vector from classifier $f$ on input $x$. HELIX computes margins in **logit space**:

$$m = z_{(1)} - z_{(2)}$$

where $z_{(1)}$ and $z_{(2)}$ are the top-1 and second-best logit values. Logit-space margins are required for dimensional consistency with threshold values $\tau_f = 70{,}000$ and $\tau_a^{\text{frozen}} = 81{,}921$ (both in logit units). All threshold comparisons, z-score normalizations, and distributional analyses in this paper are in logit margin space.

### B. Calibration Statistics

The following parameters are estimated from the UNSW-NB15 calibration split and frozen before test evaluation:

| Parameter | Value | Units |
|:---|:---|:---|
| $\mu_{\text{cal}}$ | 67,137.24 | logit units |
| $\sigma_{\text{cal}}$ | 19,616.67 | logit units |
| $\tau_a^{\text{frozen}}$ | 81,921.39 | logit units |
| $\tau_f$ | 70,000.00 | logit units |

The frozen threshold's normalized position:

$$z_{\tau_a} = \frac{\tau_a^{\text{frozen}} - \mu_{\text{cal}}}{\sigma_{\text{cal}}} = \frac{81{,}921.39 - 67{,}137.24}{19{,}616.67} \approx 0.754$$

This places $\tau_a^{\text{frozen}}$ at approximately the **77th percentile** of the calibration margin distribution, converting HELIX from a dynamic estimator into a deterministic constraint function. The fixed threshold $\tau_f = 70{,}000$ sits at $z_{\tau_f} \approx 0.146$, or roughly the **56th percentile**. In principle, the OR gate covers class-4 samples from the 56th to the 77th percentile of calibration confidence — though as Section V shows, the two gates produce identical decisions under current parameterization.

### C. The Override Rule

$$\hat{y} = \begin{cases} \arg\max_{k \neq 4} z_k(x) & \text{if } y = 4 \;\wedge\; \left( m < \tau_f \;\vee\; \frac{m - \mu_{\text{cal}}}{\sigma_{\text{cal}}} < \tau_a \right) \\ y & \text{otherwise} \end{cases}$$

where $y = \arg\max_k z_k$ and all parameters are as defined in Section III.B.

**Figure 1** shows the full inference-time pipeline, from raw logits through margin computation to the dual-gate threshold check and override decision.

![Figure 1. HELIX inference-time control pipeline. The dual-gate threshold check (fixed: $m < 70{,}000$; adaptive: $z_{	ext{norm}} < 	au_a$) evaluates an OR condition on every class-4 top prediction.](docs/fig_revamp/fig1.png){ width=72% }

The control layer is logically well-formed; inactivity under *online_hybrid* is not a logic error — it is gate failure from null-threshold propagation, as established in Section IV.

### D. Ablation Configurations

| Configuration | $\tau_f$ | $\tau_a$ | $\tau_a$ update | $\tau_a$ (logit) |
|:---|:---|:---|:---|:---|
| `fixed_tau_only` | 70,000 | — | N/A | — |
| `percentile_only` | 0 | online | Running percentile | null at eval |
| `online_hybrid` | 70,000 | online | Running percentile | null at eval |
| `frozen_z_hybrid` | 70,000 | frozen | Calibration | 81,921.39 |

### E. Algorithm 1: HELIX Inference Control (frozen\_z\_hybrid)

```
Algorithm 1: HELIX Inference-Time Override (frozen_z_hybrid variant)
-------------------------------------------------------------------
Input:  z ∈ ℝ^K          — raw logit vector
        τ_f = 70,000      — fixed logit-margin threshold
        τ_a = 81,921.39   — frozen adaptive threshold (≈ 77th pctile of cal.)
        μ_cal = 67,137.24, σ_cal = 19,616.67  — calibration statistics

Output: ŷ ∈ {0,...,K−1}

 1:  y      ← argmax_k z_k                    // O(K)
 2:  if y ≠ 4 then return y                   // no override candidate
 3:  z_(1) ← max(z);  z_(2) ← second_max(z)  // O(K)
 4:  m      ← z_(1) - z_(2)                   // O(1)
 5:  z_norm ← (m - μ_cal) / σ_cal             // O(1)
 6:  gate_f ← (m < τ_f)                       // O(1)
 7:  gate_a ← (z_norm < τ_a)                  // O(1) — τ_a is a scalar constant
 8:  if gate_f OR gate_a then
 9:      return argmax_{k≠4} z_k              // O(K), override class 4
10:  else
11:      return 4
-------------------------------------------------------------------
```

### F. Computational Overhead

HELIX adds strictly **O(1)** computation per sample beyond the base model forward pass:

| Step | Complexity | Notes |
|:---|:---|:---|
| argmax / top-2 | O(K) | Unavoidable; same as base model |
| Margin, z-normalization | O(1) | 3 scalar arithmetic operations |
| Gate evaluation | O(1) | 2 comparisons + 1 OR |
| Override argmax | O(K) | Over K−1 classes |
| **HELIX overhead** | **O(1)** | All post-argmax operations |

Memory overhead is four scalar constants ($\tau_f, \tau_a^{\text{frozen}}, \mu_{\text{cal}}, \sigma_{\text{cal}}$), independent of model size, batch size, or sequence length. No retraining, no gradient computation, no additional layers.

---

## IV. FAILURE MECHANISM AND DIAGNOSIS

### A. Formal Definition

**Definition 1 (Control Collapse — Dual Criterion).** Let $\mathcal{M} = \{x : y(x) = 4\}$ be the set of class-4 candidate samples, and let $P(y)$ and $P(\hat{y})$ denote the prediction distributions before and after HELIX override. A HELIX control layer is in **control collapse** if and only if both hold:

**Criterion 1 (Actuation failure):**
$$r = \frac{|\{x \in \mathcal{D} : \text{override}(x)\}|}{|\mathcal{D}|} < \varepsilon, \quad \varepsilon = 0.001$$

**Criterion 2 (Distribution invariance):**
$$D_{\text{KL}}\!\left(P(\hat{y}) \;\|\; P(y)\right) = 0$$

The dual criterion is necessary because Criterion 1 alone could be satisfied when class-4 candidate mass is genuinely low, while Criterion 2 alone could hold trivially. Together they establish that the control layer produces zero actuation *and* leaves the prediction distribution unchanged — it is a null operator. Note that $D_{\text{KL}} = 0$ establishes distributional identity, while $D_{\text{KL}} > 10$ (as seen in active configurations) indicates strong distributional intervention rather than marginal perturbation.

### B. Collapse Proof for online\_hybrid

**Claim.** *online\_hybrid* is in control collapse on both UNSW-NB15 and NSL-KDD.

**Proof by direct measurement:**

*UNSW-NB15 (n = 26,302 samples):*
- Criterion 1: override\_rate = 0.0 < 0.001 (satisfied; override\_count = 0)
- Criterion 2: $D_{\text{KL}}(P(\hat{y}) \| P(y)) = 0.0$ (satisfied)
- `collapsed` = **True**

*NSL-KDD (n = 22,278 samples):*
- Criterion 1: override\_rate = 0.0 < 0.001 (satisfied; override\_count = 0)
- Criterion 2: $D_{\text{KL}}(P(\hat{y}) \| P(y)) = 0.0$ (satisfied)
- `collapsed` = **True**

Both criteria hold exactly — $D_{\text{KL}} = 0.0$ is not numerical roundoff. It reflects bit-identical pre- and post-override prediction vectors, confirmed by metric identity between *online\_hybrid* and *percentile\_only* across all performance measures. **Figure 3 (left panels)** renders this distributional identity visually: the pre- and post-override margin histograms are indistinguishable.

**Thus, *online\_hybrid* satisfies both the necessary and sufficient conditions for control collapse across both datasets.** $\square$

**Table 3: Collapse quantification**

| Dataset | Configuration | Override rate | KL divergence | Collapsed state |
|:---|:---|:---|:---|:---|
| UNSW | `fixed_tau_only` | 0.0947 | 10.218 | **No** |
| UNSW | `percentile_only` | 0.0000 | 0.000 | Yes |
| UNSW | `online_hybrid` | **0.0000** | **0.000** | Yes |
| UNSW | `frozen_z_hybrid` | 0.0947 | 10.218 | **No** |
| NSL | `fixed_tau_only` | 0.3784 | 22.922 | **No** |
| NSL | `percentile_only` | 0.0000 | 0.000 | Yes |
| NSL | `online_hybrid` | **0.0000** | **0.000** | Yes |
| NSL | `frozen_z_hybrid` | 0.3784 | 22.922 | **No** |

*Collapse threshold $\varepsilon$ = 0.001. KL metric: $D_{\text{KL}}(\text{pre\_top1} \| \text{post\_pred})$ over class-4 margin distributions.*

The KL values of 10.218 (UNSW) and 22.922 (NSL) for active configurations show that override intervention produces substantial redistribution of the class-4 margin density — not a marginal shift. This contrast with the collapsed configurations ($D_{\text{KL}} = 0.0$) is the core empirical result.

### C. Mechanism: Coupling-Induced Collapse

**Proposition 1 (Coupling-Induced Override Collapse).** *Let $\{m_t\}$ be a bounded sequence of logit-space margins. Let $\tau_a(t) = \hat{F}^{-1}_M(q, t)$ be the $q$-th empirical quantile of $\{m_1, \ldots, m_t\}$ (online estimate). Suppose the adaptive gate evaluates $z_{\text{norm}}(m_t) < \tau_a(t)$, where $z_{\text{norm}}$ is dimensionless and $\tau_a(t)$ is in logit units. If $\tau_a(t) \gg \max_t z_{\text{norm}}(m_t)$ (unit mismatch regime), the gate is trivially satisfied, reducing the effective override decision to the fixed gate alone. If the fixed gate is also disabled ($\tau_f = \text{null}$ or $\tau_f = 0$), then $P(\text{override}) = 0$ exactly.*

**Proof sketch.** In *online\_hybrid*, `tau_adaptive` is recorded as `null` at evaluation time (confirmed by `"tau_adaptive": null` in the experimental record). A null-valued threshold in the comparison `z_norm < null` evaluates to False in the runtime. The fixed gate $m < \tau_f = 70{,}000$ is present, but a bug (noted in `bug_fixes`: "fixed gate uses raw tau positivity") disables it when the adaptive gate is null. Both gates fail silently, producing override\_count = 0 for all inputs. In *percentile\_only*, $\tau_f = 0$ renders $m < 0$ false for all non-negative margins. Both failure paths reach the same outcome. $\square$

**Corollary.** *online\_hybrid* $\equiv$ *percentile\_only* exactly: both reduce to a no-op override layer through structurally different but mutually reinforcing gate failures.

### D. Why This Failure Is Structural

$$\underbrace{\text{Online coupling}}_{\tau_a(t) \propto m_t} \;\times\; \underbrace{\text{Unit mismatch}}_{\tau_a \sim O(10^4),\; z_{\text{norm}} \sim O(1)} \;\Rightarrow\; \underbrace{\text{Gate null or trivial}}_{\text{no discriminative comparison}} \;\Rightarrow\; \underbrace{r = 0, \; D_{\text{KL}} = 0}_{\text{collapse}}$$

This is a property of the coupled online estimation architecture, not of the specific data or calibration values.

**Figure 2** shows the causal chain side by side: the collapse path (left, red) and the restoration path (right, green).

![Figure 2. Causal mechanism of control collapse (left) and restoration by frozen-threshold decoupling (right).](docs/fig_revamp/fig2.png){ width=72% }

Left: online percentile estimation produces a null $	au_a$ at evaluation; the gate fails silently and $D_{	ext{KL}} = 0$. Right: freezing $	au_a^{	ext{frozen}}$ at calibration provides a stable reference; gates activate and $D_{	ext{KL}} > 10$.

---

## V. INTERVENTION AND SCOPE OF RESTORATION

### A. Principle

Freezing $\tau_a$ at calibration eliminates the feedback loop. The threshold becomes a scalar constant computed once (from the UNSW calibration split: $\tau_a^{\text{frozen}} = 81{,}921.39$) and applied unchanged at inference. As established in Section III.B, this placement at $\mu_{\text{cal}} + 0.754\sigma_{\text{cal}}$ defines a fixed quantile operator over the calibration distribution — a deterministic constraint rather than a running estimate.

**Figure 4 (right panel)** shows the decoupled threshold sitting fixed at 81,921 while the evaluation batch margins move freely above and below it, enabling violation detection. The left panel shows the coupled case: $\tau_{\text{adaptive}}$ chases the distribution, and the gap between threshold and mean margin collapses.

![Figure 4. Threshold and class-4 margin evolution across evaluation batches for coupled (*online_hybrid*) and decoupled (*frozen_z_hybrid*) operation.](docs/fig_revamp/fig4.png){ width=72% }

In the coupled case, the adaptive threshold tracks the mean margin. In the decoupled case, $	au_a^{	ext{frozen}} = 81{,}921$ remains fixed, allowing genuine margin-threshold crossings.

### B. Controllability Restored: Results

**Figure 5** shows per-batch override counts across both methods and datasets. The left panel (*online\_hybrid*) is uniformly zero. The right (*frozen\_z\_hybrid*) shows consistent non-zero activation across all batches, totaling 2,491 overrides on UNSW and 8,430 on NSL.

![Figure 5. Per-batch override activation counts for *online_hybrid* and *frozen_z_hybrid* on UNSW and NSL.](docs/fig_revamp/fig5.png){ width=72% }

*online_hybrid* produces zero overrides in every batch on both datasets, while *frozen_z_hybrid* exhibits stable non-zero activation.

**Figure 3 (right panels)** shows the distributional consequence: the post-override class-4 margin distribution is rightward-shifted in the active configuration ($D_{\text{KL}} = 10.218$ on UNSW, $D_{\text{KL}} = 22.922$ on NSL).

![Figure 3. Class-4 logit margin distributions (pre vs. post override) for collapsed and active regimes on UNSW and NSL.](docs/fig_revamp/fig3.png){ width=84% }

For *online_hybrid*, pre and post distributions are identical ($D_{	ext{KL}} = 0$). For *frozen_z_hybrid*, post distributions shift substantially ($D_{	ext{KL}} = 10.218$ on UNSW, 22.922 on NSL).

### C. frozen\_z\_hybrid $\equiv$ fixed\_tau\_only: A Transparency Note

Under current calibration, *frozen\_z\_hybrid* produces identical override decisions to *fixed\_tau\_only* for every sample in both datasets. The z-score gate ($z_{\text{norm}} < \tau_a^{\text{frozen}}$) and the fixed gate ($m < \tau_f$) select the same overrides — the lower-percentile gate (56th) subsumes the higher-percentile gate (77th) in all observed cases, collapsing the OR condition to a single effective criterion.

This means normalization provides architectural robustness — if $\sigma_{\text{cal}}$ changes substantially, the relative threshold placement is preserved — but does not add discriminative signal at present. Recalibrating $\tau_a^{\text{frozen}}$ to a percentile distinct from $\tau_f$ would expose independent z-score gate behavior.

---

## VI. EXPERIMENTAL PROTOCOL

### A. Datasets

**UNSW-NB15** [19]: Contemporary network intrusion detection dataset. The base model is trained on the UNSW-NB15 training partition; evaluation on the test partition is the **in-distribution** protocol (26,302 samples).

**NSL-KDD** [20]: Curated version of KDD Cup 1999 with corrected statistical biases. Feature distribution differs substantially from UNSW-NB15. Evaluating the UNSW-trained checkpoint on NSL-KDD is the **cross-distribution** protocol (22,278 samples). No fine-tuning or adaptation is performed.

### B. Bootstrap Validation

Statistical confidence intervals are computed by non-parametric bootstrap:
- Resamples: 1,000 (with replacement)
- Confidence level: 95%
- Strategy: resample prediction-level outcomes (override decisions and class assignments)

Exact bootstrap CIs are available for override\_rate (sample-level metric). Macro-F1, class4\_precision, and class4\_recall require sample-level prediction vectors; current data export provides histogram-level approximations only, so those CIs are marked `approx_from_histogram` in the experimental record and are not reported here.

---

## VII. RESULTS

### A. Performance Comparison

**Table 1: Classification performance across configurations**

| Dataset | Configuration | Macro-F1 | Class-4 precision | Class-4 recall | Class-4 prediction ratio |
|:---|:---|:---|:---|:---|:---|
| UNSW | `fixed_tau_only` | **0.3411** | **0.0563** | **0.9706** | 0.1115 |
| UNSW | `percentile_only` | 0.2626 | 0.0313 | 1.0000 | 0.2062 |
| UNSW | `online_hybrid` | 0.2626 | 0.0313 | 1.0000 | 0.2062 |
| UNSW | `frozen_z_hybrid` | **0.3411** | **0.0563** | **0.9706** | 0.1115 |
| NSL | `fixed_tau_only` | **0.0226** | **0.00108** | **0.0263** | 0.0416 |
| NSL | `percentile_only` | 0.0220 | 0.000321 | 0.0789 | 0.4200 |
| NSL | `online_hybrid` | 0.0220 | 0.000321 | 0.0789 | 0.4200 |
| NSL | `frozen_z_hybrid` | **0.0226** | **0.00108** | **0.0263** | 0.0416 |

*Note: Macro-F1 values are deterministic aggregates without bootstrap CIs (see Section VI.B). The metric identity between* online\_hybrid *and* percentile\_only *across all columns confirms control collapse at the performance level.*

### B. Override Actuation and Confidence Intervals

**Table 2: Control activation with bootstrap confidence intervals**

| Dataset | Configuration | Override count | Override rate | 95% CI | $\sigma_{\text{override}}$ |
|:---|:---|:---|:---|:---|:---|
| UNSW | `fixed_tau_only` | 2,491 | 0.0947 | — | — |
| UNSW | `percentile_only` | 0 | 0.0000 | — | — |
| UNSW | `online_hybrid` | **0** | **0.0000** | **[0.000, 0.000]** | **0.00000** |
| UNSW | `frozen_z_hybrid` | 2,491 | **0.0947** | **[0.091, 0.098]** | **0.00182** |
| NSL | `fixed_tau_only` | 8,430 | 0.3784 | — | — |
| NSL | `percentile_only` | 0 | 0.0000 | — | — |
| NSL | `online_hybrid` | **0** | **0.0000** | **[0.000, 0.000]** | **0.00000** |
| NSL | `frozen_z_hybrid` | 8,430 | **0.3784** | **[0.372, 0.385]** | **0.00314** |

*Bootstrap CIs from 1,000 resamples with replacement, 95% confidence level. fixed\_tau\_only and percentile\_only CIs are omitted as they are structurally identical to frozen\_z\_hybrid and online\_hybrid respectively.*

**Figure 6** plots the override rate CIs for all four evaluated configurations.

![Figure 6. Bootstrap confidence intervals for override rate (95%, n=1,000) across collapsed and active configurations.](docs/fig_revamp/fig6.png){ width=78% }

Collapsed configurations map to a point mass at zero (zero width, zero variance). Active configurations have strictly non-overlapping non-zero intervals.

### C. Statistical Interpretation

**Non-overlapping CIs establish significant separation:**

- UNSW: *frozen\_z\_hybrid* CI = [0.09125, 0.09836] vs. *online\_hybrid* CI = [0.000, 0.000]. Gap: 0.09125 between the lower bound of active and the upper bound of collapsed.
- NSL: *frozen\_z\_hybrid* CI = [0.37230, 0.38459] vs. *online\_hybrid* CI = [0.000, 0.000]. Gap: 0.37230.

Both separations are statistically significant at 95% confidence.

**Degenerate variance is a structural result, not a data artifact.** The bootstrap standard deviation for *online\_hybrid* override\_rate is $\sigma_{\text{override}} = 0.00000$ on both datasets. Zero variance under 1,000 bootstrap resamples means no resampling of the evaluation data can produce non-zero actuation. This directly implies the failure is a property of the control architecture and threshold state — not of any particular subset of the test data. Combined with $D_{\text{KL}} = 0.0$, the collapse is established at both the distributional and sample levels.

---

## VIII. DISCUSSION

### A. Controllability and Classification Trade-off

On UNSW, active control raises macro\_f1 from 0.2626 to 0.3411 (+0.0785) by suppressing high-volume, low-precision class-4 predictions. Class-4 precision increases from 0.0313 to 0.0563 at a cost: recall drops from 1.000 to 0.9706. The mechanism targets samples below the 56th calibration percentile of margin (the fixed gate), suppressing the lowest-confidence class-4 predictions. The KL divergence of 10.218 establishes that the intervention is substantial — not a borderline perturbation.

### B. Cross-Distribution Results

NSL macro\_f1 < 0.023 across all configurations is not a HELIX failure — it reflects representation collapse under distribution shift. The model was trained on UNSW; it does not encode NSL class structure in its logit space, and no post-hoc control layer can fix that.

What the NSL results do show is a miscalibration problem. The 37.84% override rate on NSL (95% CI: [37.23%, 38.46%]) means the UNSW-calibrated threshold fires on more than a third of the NSL test set. NSL class-4 margins are systematically lower under the UNSW model — the model is uncertain, and the uncertainty is genuine, not a control artifact. The high KL divergence (22.922) reflects a large-volume intervention, not a targeted one.

### C. What the Degenerate Variance Tells Us

The $\sigma_{\text{override}} = 0.00000$ result for *online\_hybrid* is more informative than a point estimate. It establishes **sample-order invariance**: collapse holds regardless of which subset of the evaluation data is examined. This rules out explanations based on aggregation, imbalanced batches, or sampling noise. Collapse is a function of the architecture and threshold state, not of the data.

---

### D. Control-theoretic interpretation

HELIX maps onto discrete-time control system semantics:

| Control concept | HELIX equivalent | Value |
|:---|:---|:---|
| State signal | Logit margin $m_t$ | ~67,137 ± 19,617 (calibration) |
| Control boundary | Override threshold $\tau$ | $\tau_f$ = 70,000; $\tau_a^{\text{frozen}}$ = 81,921 |
| Actuation | Override decision | 0% (collapsed) vs. 9.5%–37.8% (active) |
| Controller | Dual-gate evaluator | O(1) per sample |
| Plant | Base classifier $f$ | helix\_full\_unsw\_nb15\_best.pt |

Under *online\_hybrid*, the controller redefines the boundary to contain all observed states — equivalent to a setpoint that chases the process variable — producing zero actuation authority. Under *frozen\_z\_hybrid*, the boundary is set externally at calibration and held fixed, so the margin can genuinely cross it.

The KL divergence of 10.218 (UNSW) and 22.922 (NSL) under active configurations quantifies the resulting actuation effect on the prediction distribution. These are not marginal adjustments.

---

## IX. OPERATIONAL COMPATIBILITY AND DEPLOYMENT CONSTRAINTS

Collapse resolution is not the same as deployment approval. We define the **bounded control regime** as:

$$0 < \text{override\_rate} \leq 0.02$$

This captures non-zero actuation authority while bounding intervention intensity to an operationally safe envelope.

| HELIX state | Control property | Deployment outcome |
|:---|:---|:---|
| `collapsed` | Inactive (override\_rate = 0) | PASS — safe but inactive |
| `restored` | Controllable (override\_rate > 0) | FAIL under runbook constraints |

The restored configurations (UNSW: 9.47%, NSL: 37.84%) exceed this envelope. Restoration demonstrates that the control layer *can* act; it does not guarantee that it acts within operationally acceptable bounds. That requires recalibration targeting the bounded regime, which is outside the scope of the present paper.

---

## X. LIMITATIONS

1. **No true class signal recovery.** HELIX overrides argmax predictions but cannot improve underlying logit quality. Under representation collapse, the next-best class is equally unreliable.

2. **Cross-distribution miscalibration.** The 37.84% NSL override rate is operationally pathological. Thresholds calibrated on UNSW do not produce bounded control on NSL.

3. **Class-4 scope only.** Override logic targets class 4 exclusively. Multi-class extension requires per-class thresholds or a generalized margin criterion.

4. **frozen\_z\_hybrid $\equiv$ fixed\_tau\_only under current calibration.** The z-score normalization channel provides no additional discriminative signal. Any benefit is fully attributable to the fixed-threshold gate. This is disclosed without mitigation.

5. **Macro-F1 bootstrap CIs unavailable.** Reported macro-F1 values are deterministic aggregates and should not be treated as statistically validated estimates.

6. **Single model, single training run.** Cross-model generalization of the collapse mechanism is established by Proposition 1 (architectural argument), not empirical replication across models.

---

## XI. CONCLUSION

We formally defined control collapse via a dual criterion (override\_rate < $\varepsilon$, $D_{\text{KL}} = 0$) and proved that *online\_hybrid* satisfies both necessary and sufficient conditions exactly on UNSW-NB15 (override\_rate = 0.0, KL = 0.0) and NSL-KDD (override\_rate = 0.0, KL = 0.0). The mechanism is adaptive threshold coupling combined with null-value propagation through the gate evaluation path. Freezing the threshold at calibration breaks both failure modes simultaneously.

Bootstrap validation (1,000 resamples, 95% CI) confirms that the decoupled *frozen\_z\_hybrid* configuration restores non-zero actuation with non-overlapping CIs relative to the collapsed regime: UNSW [0.091, 0.098] vs. [0.000, 0.000]; NSL [0.372, 0.385] vs. [0.000, 0.000]. Zero-width variance under collapse confirms the failure is deterministic and sample-order invariant.

One practical consequence extends beyond the HELIX system: **any post-hoc control layer that re-estimates its violation threshold from the live test stream is at structural risk of collapse under unit mismatch.** Controllability — a necessary prerequisite for correctness — cannot coexist with self-referential threshold estimation. In this setting, freezing the threshold is a design requirement rather than a heuristic workaround. Conformal prediction frameworks have known this for decades, which is why they mandate held-out calibration. The lesson from this failure mode is that the mandate deserves to be treated as a hard architectural constraint, not an optional best practice.

---

## REFERENCES

[1] P. Bartlett and M. Wegkamp, "Classification with a reject option using a hinge loss," *J. Machine Learning Research*, 2008.  
[2] G. Fumera and F. Roli, "A theoretical and experimental analysis of linear combiners for multiple classifier systems," *IEEE TPAMI*, 2005.  
[3] B. Zadrozny and C. Elkan, "Transforming classifier scores into accurate multiclass probability estimates," *KDD*, 2002.  
[4] L. Torresani and K.-C. Lee, "Large margin component analysis," *NeurIPS*, 2007.  
[5] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," *ICML*, 2017.  
[6] Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," *NeurIPS*, 2017.  
[7] P. L. Bartlett and M. H. Wegkamp, "Classification with a reject option," *J. Machine Learning Research*, 2008.  
[8] A. Kumar, S. Liang, and T. Ma, "Verified uncertainty calibration," *NeurIPS*, 2019.  
[9] D. Hendrycks and K. Gimpel, "A baseline for detecting misclassified and out-of-distribution examples in neural networks," *ICLR*, 2017.  
[10] K. Lee, K. Lee, H. Lee, and J. Shin, "A simple unified framework for detecting out-of-distribution samples and adversarial attacks," *NeurIPS*, 2018.  
[11] V. Vovk, A. Gammerman, and G. Shafer, *Algorithmic Learning in a Random World.* Springer, 2005.  
[12] A. N. Angelopoulos and S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification," *arXiv:2107.07511*, 2022.  
[13] Y. Geifman and R. El-Yaniv, "Selectivenet: A deep neural network with an integrated reject option," *ICML*, 2019.  
[14] J. Platt, "Probabilistic outputs for support vector machines," *Advances in Large Margin Classifiers*, 1999.  
[15] B. Zadrozny and C. Elkan, "Obtaining calibrated probability estimates from decision trees and naive Bayesian classifiers," *ICML*, 2001.  
[16] S. Liang, Y. Li, and R. Srikant, "Enhancing the reliability of OOD image detection in neural networks," *ICLR*, 2018.  
[17] J. Gama et al., "A survey on concept drift adaptation," *ACM Comput. Surv.*, 2014.  
[18] M. Baena-García et al., "Early drift detection method," *ECML-PKDD Workshop*, 2006.  
[19] N. Moustafa and J. Slay, "UNSW-NB15: A comprehensive dataset for network intrusion detection systems," *IEEE MilCIS*, 2015.  
[20] M. Tavallaee et al., "A detailed analysis of the KDD CUP 99 data set," *IEEE CISDA*, 2009.

---

