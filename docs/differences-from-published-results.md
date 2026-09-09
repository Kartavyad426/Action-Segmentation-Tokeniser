# Differences From Published Reference Results

Living document — every place our implementation might diverge from what Nomadic AI's blog
and the M2R2 paper report, or from our own spec's intent, in ways that could affect whether
our F1@50 numbers are actually comparable to theirs. Report everything here, minor or major
— this doc is for surfacing differences, not deciding whether they matter. Update whenever
a new one is found (e.g. during code review) or an existing one is resolved.

Status legend: 🔴 open / affects result validity — 🟡 open / minor or unclear impact —
🟢 resolved.

## Reference numbers we're comparing against

- **Nomadic AI blog** ("Segmenting the Key Micro-Actions in Robotic Footage"): 93.1% F1@50
  on REASSEMBLE, vs. M2R2's 83.4%. Context alone (before vision) reportedly tripled a
  random-forest baseline from 25.8% to 79.5%; vision took it to 93.1%.
- **M2R2** (arXiv 2504.18662): 83.4% F1@50 on REASSEMBLE (82.4% on a different benchmark,
  REASSEMBLE-adjacent — check the paper if citing this precisely). Proprioception-alone
  ablation: 74.5%; +vision/audio: 74.6% (vision barely helped in their specific late-fusion
  design).
- **Caveat inherited from the design spec:** the Nomadic numbers are from a vendor blog
  post with no linked paper or repo found — not independently verified the way M2R2's arXiv
  result is. Treat as directional, not ground truth.

## Architecture-level differences (deliberate, made where the reference doesn't specify)

🟡 **Nomadic's blog gives no architecture specifics** — codebook size, window length,
classifier depth, exact fusion mechanism are all undisclosed. Every hyperparameter in our
implementation (512 codes, 32-dim latent, 150ms window, 3-stage/9-layer/64-channel MS-TCN)
is an engineering choice we made, not a reproduction of theirs. This means a lower number
from us doesn't necessarily mean "vision helps less here" — it could mean "our
hyperparameters are undertuned relative to whatever Nomadic used."

🟡 **M2R2 uses three modalities (vision + audio + proprioception) with late fusion and
separate pretrained encoders per modality (ActionCLIP for vision, AST for audio). We use
two (motion tokens + vision only, concat+project) and never touch REASSEMBLE's audio data
at all**, despite it being present in the dataset and part of what M2R2's ablation measures.
If audio turns out to matter for some action classes (e.g. audible contact events during
insertion), our result isn't directly comparable to M2R2's on that axis.

🟢 **We use a VQ-VAE motion tokenizer (unsupervised, discrete) as the input representation;
M2R2 uses a learned but non-discretized proprioception projection (no VQ step, no
reconstruction objective) trained end-to-end with the classifier.** This is the core
architectural difference the whole project is built around — deliberate, per the design
spec's key decisions, not a bug. Documented for completeness, not flagged as a problem.

🟢 **MS-TCN, not Transformer**, for the temporal classifier — a staged decision (start
simple, escalate only if MS-TCN can't capture enough context once measured). M2R2 also uses
MSTCN-family heads, so this axis is actually reasonably comparable.

## Implementation gaps found during review (affect result validity until resolved)

🟢 ~~**F1@50 aggregation convention mismatch.**~~ **Fixed** (final-review fix wave). Original
implementation computed `val_f1` as a macro-average of per-demo `f1_at_k` scores. The
standard TAS protocol — how Nomadic's and M2R2's numbers are actually computed — accumulates
TP/FP/FN across all videos first, then computes one corpus-level F1
(`f1_at_k_corpus` in `temporal_classifier/metrics.py`). `run_training` now uses the
corpus-level number. This was a real "not apples-to-apples" bug, not just a style
preference — macro-averaging systematically differs from (and is noisier than)
corpus-aggregation, especially with few/short validation demos.

🟢 ~~**DINOv2 fed incorrectly preprocessed frames.**~~ **Fixed** (final-review fix wave).
Original implementation did a direct `cv2.resize` to 224x224 (squashing aspect ratio) with
no ImageNet normalization — DINOv2 was trained expecting resize-shortest-side +
center-crop + `mean=[0.485,0.456,0.406]`/`std=[0.229,0.224,0.225]` normalization. Degraded
input features bias any vision-vs-no-vision comparison toward "vision doesn't help,"
directly undermining the branch's central question. Fixed in
`enrichment/vision_features.py`'s new `_preprocess_for_dinov2`.

🟢 ~~**Vision/motion embedding scale mismatch in the fusion layer.**~~ **Fixed.**
`ConcatProjectFusion` concatenated the motion embedding and the DINOv2 CLS token with no
normalization and projected 416-dim -> 32-dim. Because `nn.Linear` draws all of its input
weights from one distribution — it cannot know which columns belong to which modality —
each branch's contribution to the output scales with that branch's vector norm, so the
input scale gap *was* the branch-dominance gap, at initialization and in the gradients.

Measured before/after, on 200 real tokens from `2025-01-09-13-57-17.h5` with a tokenizer
trained on 3 real demos and real DINOv2 features:

| | motion norm | vision norm | norm ratio | per-elem RMS ratio | output contribution ratio |
|---|---|---|---|---|---|
| before | 4.30 | 46.54 | 10.8x | 3.11x | **11.2x** |
| after | 5.66 = √32 | 19.60 = √384 | **3.46x** | **1.00x** | **3.62x** |

Note the before-numbers differ from the code reviewer's original measurement (motion norm
1.83, ratio 25.7x) because the motion norm depends on how the tokenizer happened to train —
**the imbalance was not a fixed 26x but a quantity that drifts with the tokenizer
checkpoint**, which is an additional reason normalization was needed: the branch balance was
not reproducible run to run. After the fix both branch norms are pinned to √dim by
construction and are identical regardless of tokenizer state.

The fix is `nn.LayerNorm` on **each branch separately, before** the concatenation. Applying
one `LayerNorm(416)` to the concatenated vector instead divides both halves by the same
scalar and so barely moves their ratio at all — measured on the test fixture, 25.8x raw
becomes 24.3x, against 3.46x for per-branch normalization. The per-branch placement is the
load-bearing part, not the presence of a LayerNorm. The residual 3.46x is exactly √(384/32), pure
dimensionality, and the learnable affine gains can re-weight from there.

`out_dim` is also no longer pinned to `config["latent_dim"]`; it is an independent
`run_training(fusion_out_dim=...)` parameter defaulting to 128, and `MSTCN`'s `in_channels`
follows the fusion width on the vision arm. The telemetry-only control arm is unchanged and
still built at `latent_dim` — covered by a dedicated regression test.

One thing the fix does **not** do: LayerNorm centers across dims *within* one vector, so it
rescales but does not remove the component of the DINOv2 feature that is constant *across
frames* — measured at norm 43.49 of the 46.54 total, i.e. ~87% of the vision vector's
magnitude is frame-invariant, with a per-frame residual of 16.26. The projection's bias can
absorb a constant input direction, so this costs conditioning rather than correctness. If
the vision arm still underperforms, per-dimension standardization computed over the training
frames is the next escalation.

🔴 **No caching of vision features, unbounded RAM per demo.** `_extract_frames` in
`enrichment/vision_features.py` decodes and holds every requested video frame in memory
before running any of them through DINOv2 — measured ~3.5GB for a single demo's worth of
tokens (3,814 tokens on one real demo). The spec explicitly calls vision extraction "frozen,
one-time, cacheable," but nothing is cached; every `run_training(use_vision=True)` call
re-extracts and re-encodes every frame from scratch. Not a correctness bug today (it ran
successfully in testing) but a real ceiling once the full 148-demo set trains for real —
longer demos could exhaust RAM, and repeated runs waste significant GPU time re-encoding
identical frames. Needs streaming (decode -> encode -> discard) and batched encoder calls
before the real benchmark run, plus a disk cache keyed by (demo, camera, window config).

🔴 **`boundary_alignment_score` has no chance-baseline or precision term.** It currently
measures only "fraction of ground-truth boundaries with *some* nearby token change" — pure
recall, no penalty for a tokenizer that changes token on every single window (which would
score a perfect 1.0 despite being useless). Measured on real data with a lightly-trained
tokenizer: score 0.371, but the token-change rate was high enough that the chance-level
score for random boundary placement is ~0.40 — i.e., the measured score was *at or below
chance*, and the function has no way to report that. This is one of the spec's three
tokenizer-quality gating checks; as built, it can't actually gate anything. Needs a
chance-baseline comparison and a within-segment token-stability (purity) term, which the
spec asked for but wasn't implemented.

🔴 **The tokenizer's own gating checks are never run before classifier training.**
`tokenizer/evaluate.py`'s three checks (reconstruction error, codebook utilization,
boundary alignment) exist and are unit-tested, but `temporal_classifier/compare.py`'s
`main()` never calls them — it goes straight from `train_tokenizer` to `run_training`. The
spec is explicit that these checks should *gate* moving on to the classifier. Measured
directly: a 3-epoch tokenizer trained during review had codebook utilization 0.235, using
only 9 of 512 codes — a real codebook collapse — and nothing in the current pipeline would
have caught it before it silently propagated into classifier training. Needs wiring into
`compare.py` before the real benchmark run.

## Minor implementation notes (unlikely to affect result validity, tracked for completeness)

🟡 Checkpoint config doesn't record telemetry channel names/order or the resample rate —
if a future caller passes a different channel list to `load_demo`, the saved mean/std would
silently apply to a differently-ordered feature vector.

🟡 `f1_at_k` returns `0.0` (not skip/NaN) for a demo with zero non-background ground-truth
segments — a demo with no labeled action drags any average down as if it were a total miss,
rather than being excluded.

🟡 No class-imbalance handling in the classifier's `CrossEntropyLoss` — background is the
dominant class in real data (measured ~46% of tokens on one demo). Unclear whether
Nomadic/M2R2 handle this; worth checking if our numbers come in surprisingly low.

🟡 `train_tokenizer` has no per-epoch logging or held-out validation loss — `final_loss` is
just the last training batch's loss (noisy), not a real training-quality summary. Fine for
smoke tests, insufficient for judging convergence on the real 148-demo run.

## How to use this doc

Before citing any F1@50 number from this pipeline against Nomadic's or M2R2's, check this
doc's 🔴 items — those are the ones most likely to make a direct comparison misleading.
When a 🔴 or 🟡 item gets fixed, move it under a `~~strikethrough~~` note (see the F1
aggregation and DINOv2 preprocessing entries above) rather than deleting it — the history of
what was wrong and when it got fixed is itself useful context for interpreting any numbers
produced before the fix.
