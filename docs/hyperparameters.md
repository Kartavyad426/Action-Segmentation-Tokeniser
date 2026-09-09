# Hyperparameters

Every hyperparameter in this pipeline, where it is set, and which ones are worth changing.

**None of these are reproductions of a published configuration.** Nomadic's blog discloses no
architecture specifics and M2R2's design differs structurally (see
`docs/differences-from-published-results.md`), so every value below is an engineering choice
we made. A lower number from us does not necessarily mean "vision helps less here" — it could
mean these are undertuned.

Tune against the **validation** split only (`split_available_demos_3way`, default
`--eval-on val`). `test_split1` is REASSEMBLE's published held-out set; anything tuned
against it stops being comparable to published numbers.

---

## Tokenizer

Set in `tokenizer/train.py::train_tokenizer` and `tokenizer/model.py::MotionTokenizer`.

| param | default | where | note |
|---|---|---|---|
| `num_codes` | 512 | `MotionTokenizer` | **measured utilization 0.235 — only 9 codes live** |
| `commitment_cost` | 0.25 | `_VectorQuantizer` | drives codebook collapse behaviour |
| `latent_dim` | 32 | `MotionTokenizer` | now safely sweepable (decoupled from fusion/classifier) |
| `window` | 15 = 150 ms | `train_tokenizer` | token granularity |
| `stride` | 5 | `train_tokenizer` | training only; classifier uses non-overlapping windows |
| `hidden` | 64 | `MotionTokenizer` | encoder/decoder width |
| `epochs` | 20 | `train_tokenizer` | tuned on 3 demos, never on 111 |
| `batch_size` | 64 | `train_tokenizer` | |
| `lr` | 1e-3 | `train_tokenizer` | |

## Classifier

Set in `temporal_classifier/train.py::run_training` and `temporal_classifier/model.py::MSTCN`.

| param | default | where | note |
|---|---|---|---|
| `channels` | 64 | `MSTCN` | |
| `num_layers` | 9 | `MSTCN` | receptive field already 1,023 tokens ≈ 153 s — ample, don't grow it |
| `num_stages` | 3 | `MSTCN` | |
| `fusion_out_dim` | 128 | `run_training` | independent of `latent_dim` since the fusion fix |
| `epochs` | 30 | `run_training` | no early stopping, no val-loss tracking |
| `lr` | 1e-3 | `run_training` | |

## Vision

Set in `enrichment/vision_features.py` and threaded through `run_training`.

| param | default | where | note |
|---|---|---|---|
| `VISION_FEATURE_DIM` | 384 | module constant | fixed by `dinov2_vits14`'s CLS token |
| `batch_size` | 32 | `extract_frame_features` | DINOv2 encode batch |
| `camera_key` | `hama1` | `run_training` | `hama2` and `hand` also exist; audio unused |
| `pool_vision_over_window` | `False` | `run_training` | `True` mean-pools every frame in the token's window instead of taking the nearest one |
| `cache_dir` | `None` | `extract_frame_features` | `compare.py` uses `vision_cache/` |

---

## The three worth changing

### 1. `num_codes` 512 → 64 or 128, and/or tune `commitment_cost`

Measured codebook utilization is **0.235 — 9 of 512 codes actually in use**. That means the
motion representation is effectively a 9-symbol alphabet: the classifier is being starved
before vision ever enters the picture. This is the most broken measured number in the
pipeline and plausibly the single largest F1 lever.

Shrinking the codebook is the cheap fix. The proper fixes are EMA codebook updates or
dead-code restarts.

### 2. Add the T-MSE smoothing loss

The classifier's loss is plain `CrossEntropyLoss` summed across stages
(`temporal_classifier/train.py:117,133`). MS-TCN's signature contribution is a **truncated
MSE smoothing loss** over adjacent-frame log-probabilities — typically `L = CE + 0.15·T-MSE`
with τ=4 — and we do not have it.

Its entire purpose is suppressing over-segmentation, which is exactly what F1@50 punishes: a
fragmented prediction generates many false-positive segments that fail the 0.5 IoU match.
Expect this to move F1@50 more than any class-imbalance treatment. Its absence is also a
comparability gap, since MSTCN-family heads normally include it.

### 3. Tokenizer `epochs`

20 was chosen against 3 demos. 111 demos is ~37x the data, with no held-out loss tracking to
say whether it is converged or overfit — `final_loss` is just the last training batch (an
open 🟡 in the differences doc).

Everything else should be left alone for the first run.

---

## Class imbalance: do not resample

Measured low-level label counts across the demos on disk:

| label | segments (train) |
|---|---|
| Grasp | 1,598 |
| Approach | 1,543 |
| Release | 1,269 |
| Lift | 814 |
| Align | 806 |
| Pull | 676 |
| Twist | 189 |
| Push | 181 |
| Nudge | 55 |

Plus `background`, which is class 0. **10 classes total.**

Token-level distribution, measured through `align_labels_to_grid` on a 12-demo sample per
split (this is what the classifier actually sees, and it differs from the segment counts
above because segments have very different durations):

| label | train | val | test |
|---|---|---|---|
| background | 1.8% | 1.9% | 3.4% |
| Align | 12.6% | 18.8% | 11.5% |
| Approach | 29.8% | 27.2% | 25.9% |
| Grasp | 36.5% | 33.3% | 37.4% |
| Lift | 3.1% | 3.0% | 2.3% |
| Nudge | 1.1% | 1.1% | 0.9% |
| Pull | 4.5% | 3.2% | 9.2% |
| Push | 2.7% | 4.1% | 2.4% |
| Release | 3.4% | 3.1% | 4.1% |
| Twist | 4.5% | 4.2% | 2.9% |

**Background is 1.8%, not dominant.** An earlier note recorded ~46%, taken from a single
demo; it does not hold at corpus level. The real imbalance is **33:1 between action
classes** — Grasp vs Nudge.

Note also that the splits are not distributionally identical (`Align` 12.6% train vs 18.8%
val; `Pull` 4.5% train vs 9.2% test), so validation is good for ranking configurations but
its absolute number is not a test estimate.

Resampling is the standard answer for classification and the wrong tool here. This is dense
sequence labeling: you cannot oversample a `Nudge` token without duplicating it *inside* a
sequence, which fabricates a segment that never happened and corrupts the temporal structure
the MS-TCN exists to model. Resampling at the demo level only changes which demos appear more
often; it barely moves the token-level balance and it biases the segmentation prior.

Three things to know before spending effort here:

1. **Background is already excluded from the metric.** `metrics.py:20-21` drops
   `background_label=0` segments from both prediction and ground truth before matching. So
   background does not directly cost F1@50 — and at a measured 1.8% of tokens it is not the
   problem it was assumed to be anyway. It costs only indirectly: a model that over-predicts
   background loses real segments as false negatives.
2. **The real imbalance is between action classes** — `Nudge` at 55 vs `Grasp` at 1,598 is
   29:1. That is what would actually go unlearned.
3. **Class weighting and smoothing pull in opposite directions.** Upweighting rare classes
   produces more short spurious segments → more over-segmentation → *lower* F1@50. Add the
   smoothing loss first.

**Recommended order:** add T-MSE → measure per-class F1 on validation to see whether
`Nudge`/`Twist`/`Push` are actually being missed → only then try median-frequency class
weights in `CrossEntropyLoss(weight=...)`, tuned on validation. Resampling: no.
