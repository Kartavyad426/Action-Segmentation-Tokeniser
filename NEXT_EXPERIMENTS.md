# Next experiments

Ranked backlog for versions after `results/v1`. Each entry says what to change, why, what it
costs, and what it would tell us. Ordering is by expected effect on the numbers, not by
effort.

Current state: v1 measured telemetry-only **0.8419**, vision at one frame per token
**0.7746** (-0.067), vision pooled over the token window **0.8152** (-0.027), all on the
validation split, single run, no error bar.

---

## Tier 1 — will change v1's numbers, cheap

### 1.1 Reproducibility study *(running)*

Three full runs at tokenizer/classifier seeds 1, 2, 3. Establishes the seed-to-seed spread of
the control arm, which is the noise floor the -0.027 has to clear.

Until this exists, **no delta in v1 can be called real**. If the control arm swings ±0.04
across seeds, the surviving vision effect is unmeasurable and every conclusion about pooling
needs restating as "within noise".

Cost: ~3 x tokenizer retrain. The vision cache survives a seed change (it keys on token
centres, not on tokenizer weights), so feature extraction is free.

Aggregate with `python -m temporal_classifier.summarize runs/` — reports per-arm spread plus
the delta computed *within* each run and then averaged.

### 1.2 Classifier epoch budget and early stopping

**Two of three arms were still climbing at epoch 30.** The control went 0.8211 → 0.8419 over
the final five epochs; the pooled arm 0.8005 → 0.8152. The fixed budget is truncating them,
and `run_training` reports whatever epoch 30 produced.

Three changes:

- `eval_every=1`. Measured cost of a validation pass: **~0.1 s** against a 0.8 s epoch. The
  current 6-point curve is coarse for a metric as discontinuous as F1@50, for no saving.
- Extend the epoch budget until the curves flatten, on validation.
- Track best-epoch per arm, as the tokenizer already does.

One trap: early-stopping on validation *and* reporting that same validation number is
optimistically biased. Tune the budget on validation, freeze it, report test.

A second trap, raised by a peer session: the moment early stopping exists, per-arm wall time
stops being a throughput measure, because an arm that converges early stops early. Record
epochs-completed and seconds-per-epoch **before** adding the early exit, so the normalised
number exists first.

---

## Tier 2 — improve the representation both arms depend on

### 2.1 Codebook initialization from real encoder outputs

`tokenizer/model.py` initializes with `uniform_(-1/num_codes, 1/num_codes)`. Measured at
init with 512 codes:

```
codebook vectors: mean norm 0.00636, mean inter-code distance 0.00897
encoder output z: mean norm 0.58698        <- 92x farther from the origin
distance from one z to all 512 codes: nearest 0.52901, farthest 0.53651  (1.4% spread)
distinct codes winning across 256 inputs: 2 of 512
```

**The tokenizer starts collapsed to 2 codes.** Every code sits essentially on the origin
while the data sits at radius ~0.59, so all 512 are equidistant to within noise and the same
one or two win everything. Only codes that win assignments receive gradient, so the rest stay
frozen — the classic dead-code spiral. Roughly the first 10 epochs of every run are spent
undoing this rather than learning motion (utilization climbed 0.462 → 0.733 over 20 epochs).

The scaling is also inverted: **more codes means a tighter init**, so raising `num_codes` for
diversity makes collapse *more* likely.

Fixes, cheapest first: initialize codes by sampling real encoder outputs (or k-means over
them); EMA codebook updates; dead-code restarts. Any of these should also make tokenizer
training far more reproducible than a seed alone, since whether a run passes through a
collapsed phase is currently luck.

### 2.2 Boundary precision 0.115

Only ~12% of token changes land near a real action boundary — the tokenizer carves far more
finely than the actions it is meant to track. This reproduced across runs (0.118, 0.115), so
it is a property of the configuration, not noise, and it is a ceiling on F1@50 for **every**
arm independent of vision.

Worth a window-length sweep now that `latent_dim` is decoupled from the fusion and classifier
widths. 150 ms against primitives with a median duration of ~6.8 s may simply be too fine.

### 2.3 `latent_dim` and `num_codes` sweeps

Now safely sweepable: `fusion_out_dim` is independent, so changing `latent_dim` no longer
silently changes classifier capacity too. Do 2.1 first — the current knob partly measures
init pathology rather than capacity.

---

## Tier 3 — give vision a fairer hearing

**These change the question.** v1 answers "does vision help in a simple concat+project
design". Each item below makes vision *more* likely to help, which is a different and also
legitimate question — but it should be asked deliberately, not slid into.

### 3.1 Gated or attention-based fusion

The design difference most likely to explain why ours *hurt* where M2R2's was neutral. M2R2
fuses with a modality transformer (self-attention + MLP), where attention can attenuate a
useless modality toward zero. Our concat → linear → MS-TCN keeps the vision dimensions in
every input vector with no way to gate them away.

### 3.2 Per-dimension standardization of DINOv2 features

~87% of the CLS token's magnitude is a component constant across frames (norm 43.49 of
46.54). LayerNorm rescales it but cannot remove it, because it centres across dims *within* a
vector, not across frames. Standardizing per dimension over training frames would.

### 3.3 A video-pretrained vision encoder

DINOv2 is image-level and self-supervised; its spatial features are documented as lacking
motion information for temporal tasks. M2R2 uses ActionCLIP. VideoMAEv2 is the other obvious
candidate.

### 3.4 Gradient blending / modality-balanced optimization

The literature's direct remedy for the failure v1 appears to show: modalities overfitting and
generalizing at different rates under a single optimization strategy. See
`docs/are-our-results-good.md` for references.

---

## Tier 4 — reporting and protocol

### 4.1 Test-run ledger

Every `--eval-on test` invocation appends to a committed ledger: timestamp, git commit,
fingerprint, full config, result. Contamination of a held-out set comes from *selection*, not
from looking — so if every look is on the record, a reader can judge for themselves. Without
it, "we only ran test once" is a claim nobody can check, including us.

### 4.2 Per-class F1

We do not know whether `Nudge` (0.88% of train tokens), `Twist` or `Push` are being missed
entirely. That is where the 41:1 class imbalance would show up, and it decides whether class
weighting is worth trying. Currently only corpus-level F1@50 is reported.

### 4.3 Audio

REASSEMBLE ships audio and we ignore it completely. M2R2 measured audio-only at **36.4**
F1@50 against vision-only's **21.6** — audio is the more informative exteroceptive modality
on this dataset, and we have never touched it.

### 4.4 Class weighting

Only after 1.2. Weighting rare classes produces more short spurious segments, which works
directly against the T-MSE smoothing loss. See `docs/hyperparameters.md`.

---

## The one that would settle the headline claim

**Run `--eval-on test` exactly once, with hyperparameters frozen**, after Tier 1 and ideally
2.1. That folds validation back into training (111 demos), retrains the tokenizer on all of
them, and scores the three arms on the untouched 37-demo official test split.

Until then every number in `results/` is a validation number and is not comparable to M2R2's
74.5 / 74.6 or Nomadic's claimed 79.5 / 93.1.
