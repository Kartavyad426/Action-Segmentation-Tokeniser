# Issues and Fixes

What was wrong, what was done about it, and why that solution rather than another.

`docs/differences-from-published-results.md` is the ledger of *what* diverges from published
results. This doc is the reasoning: the argument behind each fix, including the alternatives
rejected and the things each fix deliberately does **not** solve.

Every number below was measured, not estimated. Where a fix changed a measurable quantity,
the before/after is given.

---

## 1. Vision/motion embedding scale mismatch in the fusion layer — `cbde533`

**Symptom.** `ConcatProjectFusion` concatenated a 32-dim motion embedding with a 384-dim
DINOv2 CLS token and applied one `nn.Linear`, with no normalization anywhere.

**Why it mattered.** `nn.Linear` draws all 416 input weights from one distribution — it has
no idea which columns belong to which modality. So each branch's contribution to the output
scales with that branch's **vector norm**, which means the raw scale gap *is* the
branch-dominance gap, one to one, at initialization and in the gradients. The motion signal —
the thing the project is built around — entered ~11-26x quieter than vision.

This defeats the ablation the whole project exists to run. If the vision arm scores worse
than telemetry-only, you cannot distinguish "vision genuinely doesn't help here" (a
legitimate finding, consistent with M2R2's 74.5 → 74.6) from "the fusion layer was badly
conditioned" (an artifact).

**Fix.** `nn.LayerNorm` on **each branch separately, before** the concatenation. LayerNorm
pins a branch's L2 norm to exactly √d regardless of what came in, so the ratio becomes a
known constant instead of an accident of encoder scale.

Measured on 200 real tokens, real tokenizer, real DINOv2:

| | motion norm | vision norm | norm ratio | per-elem RMS ratio | output contribution |
|---|---|---|---|---|---|
| before | 4.30 | 46.54 | 10.8x | 3.11x | **11.2x** |
| after | 5.66 = √32 | 19.60 = √384 | **3.46x** | **1.00x** | **3.62x** |

**Why per-branch, and not one LayerNorm over the concatenation.** A single `LayerNorm(416)`
applies one shared scalar to both halves, so it cannot change their ratio. Verified rather
than argued: 25.8x raw becomes 24.3x, against 3.46x for per-branch. The placement is the
load-bearing part, not the presence of a LayerNorm.

**Why both branches, not vision only.** Vision-only leaves motion at per-element RMS 0.32
against vision's 1.00, so vision still enters ~10.7x louder. Normalizing both matches
per-element scale exactly and leaves 3.46x = √(384/32), which is pure dimensionality —
vision has 12x more dims — and is re-weightable through LayerNorm's learnable gains. The
cost is that per-sample LayerNorm erases the codebook vectors' magnitude differences; judged
minor, since 512 codes stay distinct under an affine map almost surely.

**An additional reason found while measuring.** The imbalance was not a fixed 26x. It drifts
with the tokenizer checkpoint — 25.7x in the original review measurement, 10.8x on a
different 3-epoch tokenizer — so the branch balance was **not reproducible run to run**.
After the fix both norms are √dim by construction, identical regardless of tokenizer state.

**What this does not fix.** LayerNorm centers across dims *within* a vector, so it rescales
but does not remove the component of the DINOv2 feature that is constant *across frames* —
measured at norm 43.49 of the 46.54 total, i.e. ~87% of the vision vector is frame-invariant,
with a per-frame residual of 16.26. The projection's bias can absorb a constant input
direction, so this costs conditioning rather than correctness. Per-dimension standardization
over the training frames is the next escalation if the vision arm underperforms.

**Also fixed here:** `out_dim` was pinned to `config["latent_dim"]`, making one tokenizer
hyperparameter silently set the fusion width *and* the classifier's input width. It is now
`run_training(fusion_out_dim=...)`, default 128. This matters beyond tidiness: with the
coupling in place, a `latent_dim` sweep would have moved the motion representation, the
fusion bottleneck, and the classifier's capacity simultaneously — confounded by construction.

---

## 2. No caching of vision features, unbounded RAM per demo — `a696068`

**Symptom.** `_extract_frames` decoded and held every requested frame in memory before
encoding any of them. Nothing was cached, so every `run_training(use_vision=True)` re-decoded
and re-encoded every frame from scratch, despite the spec calling vision extraction "frozen,
one-time, cacheable."

**Why it mattered.** Peak RAM scaled with demo length rather than anything bounded. Measured
frames are 480x640x3 = 0.88 MB each, so a 3,814-token demo holds 3.35 GB — corroborating the
~3.5 GB measured in review. Longer demos would exhaust RAM outright on the full run.

**Fix.** Three changes: `_iter_wanted_frames` streams (decode lazily, yield, discard, stop
once the last wanted frame is seen); the encoder is called once per batch of 32 instead of
once per frame; `cache_dir` persists the feature array to disk.

| frames requested | before | after |
|---|---|---|
| 300 | — | +82.9 MB |
| 600 | **+567.8 MB** | **+84.9 MB** |
| 1200 | — | +85.8 MB |

Flat: peak is O(batch size), not O(demo length).

**Why the cache key is the frame-index list**, not just the demo name: a change to the window
config moves every token center, and a demo-keyed cache would then silently serve features
aligned to the *old* token grid — a wrong-answer bug rather than a stale-data one. Keying on
the exact per-token frame grouping makes that a cache miss.

**Why the cache is not cleared between runs.** Token centers depend only on the tokenizer's
`window`, not on its weights, so cached features stay valid when the tokenizer is retrained
and self-invalidate if `window` changes. Clearing it would waste hours for no correctness
gain. This is the deliberate exception to the fresh-start policy in §4.

---

## 3. Vision sampled one frame per token, with no alternative — `10cc65c`

**Symptom.** Not a bug; an unexamined design choice. Vision is sampled down to the token rate
(6.67 Hz against a 30 Hz camera), so 680 of 3,062 frames are encoded and ~78% are discarded.

**Why it mattered.** The rate was never independently chosen — it fell out of the 150 ms
window via the 1:1 requirement of concat fusion. And the two branches are downsampled in
fundamentally different ways: the motion token *aggregates* all 15 samples in its window,
while the vision feature is a single instantaneous snapshot with the rest thrown away. That
means the vision branch is structurally limited to *what things are* and can never contribute
*what things are doing*. If the ablation returns "vision didn't help," that is a competing
explanation, and it would have been invisible.

**Fix.** Added `pool_vision_over_window=True` as a second vision arm: every frame inside the
token's window is encoded and mean-pooled, giving vision the same aggregate-don't-discard
treatment telemetry gets. `compare.py` now runs three arms.

**Why measure rather than switch.** The measured segment durations say the current sampling
is probably adequate — the shortest primitive (`Lift`, 0.95 s) still spans ~6 tokens, and on
a static camera adjacent frames are highly redundant. Rather than assume, both are now run
and the delta is data.

**Why not raise the token rate to 30 Hz instead.** Coherent, but it moves the grid rate,
window, and tokenizer capacity at once, and 33 ms of motion may be too little to tokenize
distinctly — with the codebook already collapsing at 150 ms, shorter windows push the wrong
way. Deferred until the two cheaper arms are measured.

---

## 4. No validation split; stale tokenizer checkpoints — `8c3cfd5`

**Symptom.** `split_available_demos` returned REASSEMBLE's official `train_split1` /
`test_split1`, but `compare.py` named the second one `val_paths` and reported `val_f1` from
it. Separately, `load_checkpoint` would happily reload a checkpoint left over from a smoke
test.

**Why it mattered.** Every number the pipeline produced was a **test-set** number. Fine for a
final report, fatal for tuning: anything tuned against `test_split1` stops being comparable
to M2R2 or Nomadic, which is the entire point of using the official split. And a tokenizer
trained on 3 demos silently becoming the tokenizer for a 111-demo run would apply an old
codebook and old normalization statistics to a different corpus.

**Fix.** `split_available_demos_3way` carves a validation set out of the official train half
and leaves `test_split1` untouched. `resolve_eval_split` defaults to scoring on validation
and folds val back into training only when `test` is explicitly requested, with a warning.
`reset_tokenizer_checkpoint` deletes a stale checkpoint before every run; `--keep-checkpoint`
opts out.

**Why the assignment comes from the canonical split file, not from disk.** If val membership
were computed from whatever demos happen to be present, it would change as the dataset
finished downloading — a demo held out in one run could be trained on in the next, silently.
Deriving it from `train_split1.txt` and then filtering by availability makes membership
stable at any download completeness. There is a regression test for exactly this.

**A leak found and cleared.** `run_training` builds its vocabulary from `train_paths +
val_paths`, so label text from the evaluation demos enters the setup. Scanned across all
demos on disk: all 9 low-level labels appear in both halves, so the vocabulary is identical
either way and nothing is biased today. Left as a 🟡 for hygiene rather than treated as
urgent.

---

## 5. `boundary_alignment_score` could not gate anything — `3a5b46c`

**Symptom.** It measured only "fraction of ground-truth boundaries with *some* nearby token
change" — pure recall.

**Why it mattered.** A tokenizer that emits a different code every single window hits every
boundary by accident and scores a **perfect 1.0** while carrying no information whatsoever.
Measured on real data with a lightly-trained tokenizer: 0.371, while the chance level for
randomly-placed changes at that change rate was ~0.40 — the score was *at or below chance*,
and the function had no way to say so. As one of the spec's three quality gates, it could not
gate.

**Fix.** `boundary_alignment_report` returns four things where there was one:

- `change_precision` — of the changes the tokenizer made, how many landed near a real
  boundary. This is what punishes changing constantly.
- `chance_recall` and `lift_over_chance` — what the same *number* of changes would score
  scattered uniformly at random, so the score can be read against its own null.
- `segment_purity` — within a ground-truth segment, the fraction of tokens holding that
  segment's most common code. The within-segment stability term the spec asked for and never
  got.
- `f1` — harmonic mean of recall and precision; the number to gate on.

| tokenizer behaviour | recall | chance | lift | precision | f1 | purity |
|---|---|---|---|---|---|---|
| changes every window | 1.00 | 0.93 | +0.07 | 0.06 | **0.11** | 0.05 |
| changes half the time | 1.00 | 0.74 | +0.26 | 0.04 | 0.08 | 0.10 |
| changes at boundaries only | 0.50 | 0.03 | +0.47 | 1.00 | **0.67** | 1.00 |

The old metric returned **1.00 for the first two rows**.

**Why the math was pulled out of the model.** `boundary_alignment_report(indices, centers,
segments)` is a pure function, so those three degenerate rows are unit tests instead of
something only observable on real data after a training run. Return type changed `float` →
`dict`.

---

## 6. Tokenizer gating checks never ran — `9d1f220`

**Symptom.** The three checks existed and were unit-tested, but `compare.py` went straight
from `train_tokenizer` to `run_training`.

**Why it mattered.** The 0.235 utilization / 9-live-codes collapse measured during review
would have propagated silently into hours of classifier training across three arms. The
checks were dead code.

**Fix.** `evaluate_tokenizer` runs all three on held-out demos, `check_tokenizer_gates` turns
the report into pass/fail, `compare.py` aborts before classifier training. `--skip-gates`
overrides.

**Why held-out specifically.** The spec asks for it, and the reason is real: reconstruction
error on training windows says nothing about whether the codebook captures the motion
distribution. Nothing previously evaluated on anything but training data.

**Why only one of the three checks has a threshold.** The spec says these should gate but
names no numbers. Rather than invent three:

- **Reconstruction error** — no principled absolute cutoff exists for normalized telemetry
  MSE. Only a non-finite value (diverged training) fails; the number is printed for a human.
- **Codebook utilization** — the one real threshold (0.35 normalized entropy, overridable),
  because collapse reconstructs fine and is invisible to check 1. Reported alongside
  `effective_codes` = `num_codes ** utilization`, far more legible than normalized entropy:
  the review's 0.235 reads as ~9 effective codes.
- **Boundary alignment** — fails at or below its own chance level. Threshold-free by
  construction, which §5 is what made possible.

---

## 7. "MS-TCN" in architecture only — no smoothing loss

**Symptom.** The classifier trained on plain `CrossEntropyLoss` summed across stages.

**Why we did not have it.** The spec selected MS-TCN as an *architecture* — "stacked dilated
convolutions for a large receptive field" — and the plan spelled the training loop out
verbatim, including `loss_fn = nn.CrossEntropyLoss()`. Neither document mentions smoothing,
T-MSE, or over-segmentation anywhere. The implementation was faithful to the plan; the plan
was incomplete. MS-TCN is an architecture *and* an objective, and its paper attributes a
large share of its gains to the loss rather than the stacking.

**Why it mattered.** Cross-entropy scores every token independently, so it has no objection
to a prediction that flips class between neighbouring 150 ms tokens — per-token accuracy can
be excellent. Segmental F1@50 objects strongly: that flicker becomes many short segments, at
most one of which can satisfy the 0.5 IoU match while the rest are false positives. A model
can be accurate per-token and poor at F1@50, with nothing in the objective pushing back. This
is the standard over-segmentation failure, and it is what the metric we report actually
measures.

**Fix.** `temporal_classifier/losses.py`: squared frame-to-frame difference of
log-probabilities, clamped at `tau**2` (tau=4), previous frame detached, weight 0.15, summed
over every stage. `smoothing_weight=0` reproduces the old behaviour exactly.

**Why truncation is the load-bearing detail.** A real action boundary *should* produce a
large jump in log-probabilities — that is the model doing its job. An untruncated smoothing
penalty would punish precisely the transitions we want predicted. Clamping caps what any one
frame pair can contribute, so a genuine boundary is cheap while sustained flicker — many
small penalties in a row — still accumulates into real pressure.

**Why the previous frame is detached.** The loss should pull the current frame toward the
previous one, not drag the previous frame backwards to meet it.

Both details are pinned by mutation-tested assertions: removing the clamp fails the
truncation test, removing the detach fails the gradient test.

**Ordering note.** This must land before any class weighting. Upweighting rare classes pushes
toward more short spurious segments — directly against this term.

---

## Still open

| | issue | note |
|---|---|---|
| 🟡 | `train_tokenizer` has no per-epoch or held-out loss | `final_loss` is the last training batch. Cannot distinguish converged from overfit. |
| 🟡 | No class-imbalance handling | See `docs/hyperparameters.md`. Resampling is the wrong tool; weighted CE is the option, but it works against the smoothing loss. |
| 🟡 | `f1_at_k` returns 0.0 for demos with no non-background ground truth | Drags averages down as if a total miss rather than being excluded. |
| 🟡 | Checkpoint config records no channel names/order or resample rate | Saved mean/std could silently apply to a differently-ordered feature vector. |
| 🟡 | Vocab built from train + eval | Verified harmless today (see §4). |
