# v1 — first run to produce a real answer

Source run: `runs/20260911-114547`. Raw artifacts in `data/`, figures in `images/`.
Reference targets and known divergences live in
`docs/differences-from-published-results.md`; this doc is the numbers themselves.

Numbering: `v1`, `v2`, ... one per run that produced a usable result. Aborted and failed
attempts are not numbered; they stay in `runs/` with a `status.json` saying why.

**Status: complete, all three arms.** Scored on the 22-demo **validation** split, not the
held-out test split. Nothing here is comparable to published numbers yet.

![F1@50 by arm](images/f1_comparison.png)

### Headline

| arm | F1@50 (final) | F1@50 (best) | delta (final) | delta (best) | train time |
|---|---|---|---|---|---|
| telemetry only *(control)* | **0.8419** | 0.8419 @30 | — | — | 8 min |
| vision, 1 frame/token | **0.7746** | 0.8096 @25 | **-0.0673** | -0.0323 | 58 min |
| vision, pooled over window | **0.8152** | 0.8152 @30 | **-0.0266** | -0.0266 | 108 min |

**Vision did not help, in either sampling regime.** But pooling recovers about 60% of the
deficit (-0.067 → -0.027), which is the single most informative number in this run: it says
temporal subsampling was a real and substantial part of why vision hurt, and that what
remains after fixing it is smaller and may be within noise.

Directionally consistent with M2R2's ablation on this same dataset (proprioception 74.5 →
all-modalities 74.6, vision contributing +0.1). Nowhere near Nomadic's claimed 79.5 → 93.1.

### Provenance

- git `c46aec7`, torch 2.11.0+cu128, python 3.12.13, RTX PRO 1000 Blackwell (8 GB)
- 89 train / 22 validation / 37 test demos, from REASSEMBLE's official `split1`
- tokenizer fingerprint `c0c24bcfd26cbbe6`, 20 epochs, 512 codes, 32-dim latent, 150 ms window
- classifier: MS-TCN 3 stages x 9 layers x 64 ch, 30 epochs, CE + 0.15·T-MSE
- full demo membership of all three splits is recorded in `data/config.json`, not just counts

### Provenance gaps in this run

Two fields that later runs record are missing here, because both landed *while this run was
in flight*:

- **No seed.** Seeding was added at commit `fec9981`, after this run started. Its
  `tokenizer_hp` and `classifier_hp` therefore carry no seed, and **this exact run cannot be
  reproduced** — only re-sampled from the same distribution. Runs from `fec9981` onward can.
- **No GPU power state.** The provenance capture (`pstate`, `clocks.sm`, enforced power
  limit) was added at commit `9aefa44`. This matters more than it sounds: the laptop moved
  from battery to AC *during this run*, and the GPU was clamped to 180 MHz / 15 W for part of
  it. Identical extraction work measured 601.7 s throttled and 24.0 s on AC; classifier
  epochs ran ~17 s throttled and 0.8 s on AC.

  **Consequence for this run's timings:** the per-arm `seconds` in `data/results.json` are not
  comparable to each other. The telemetry-only arm's 510 s was measured entirely inside the
  throttled window and would be roughly 20-40 s on AC. The vision arms span both regimes.
  **The F1@50 numbers are unaffected** — they are not timing-dependent.

A side effect of the seed being added mid-run: this run's tokenizer fingerprint predates it,
so resuming from it needs `--force-resume`. See `RECOVERY.md`.

### Tokenizer that produced it

![tokenizer training](images/tokenizer_training.png)

Loss and codebook utilization are plotted on twin axes deliberately: they move in *opposite*
directions. Total VQ-VAE loss falls when the codebook collapses, so the loss curve alone is
actively misleading about tokenizer quality. The dotted line is the 0.35 gate threshold.

| check | value | verdict |
|---|---|---|
| reconstruction error | 0.3548 | reported, no threshold |
| codebook utilization | 0.725 (~92 of 512 effective codes) | **pass** (floor 0.35) |
| boundary f1 | 0.192 (recall 0.598, precision 0.115) | — |
| boundary lift over chance | +0.129 (chance 0.468) | **pass** (>0) |
| segment purity | 0.501 | — |

Boundary precision of 0.115 is the weak spot and it reproduces across runs (0.118 in an
earlier one): only ~12% of token changes land near a real action boundary, so the tokenizer
carves far more finely than the actions it is meant to track. That is a plausible ceiling on
F1@50 for *both* arms, independent of vision.

---

## What the per-epoch curves say, which the headline number hides

![classifier training](images/classifier_training.png)

Validation F1@50 at each scored epoch:

| epoch | telemetry | vision nearest | vision pooled |
|---|---|---|---|
| 5 | 0.7597 | 0.6998 | 0.6809 |
| 10 | 0.8164 | 0.7604 | 0.7000 |
| 15 | 0.8225 | 0.7356 | 0.7411 |
| 20 | 0.8235 | 0.7993 | 0.7407 |
| 25 | 0.8211 | 0.8096 | 0.8005 |
| 30 | **0.8419** | 0.7746 | **0.8152** |

**The control arm is above both vision arms at every single scored epoch.** Whatever else is
uncertain, that is not a stopping-point artifact.

Three things matter here:

**1. Final training loss orders inversely to validation F1, monotonically across all three
arms.**

| arm | final train loss | val F1@50 |
|---|---|---|
| vision, 1 frame/token | **0.7913** (lowest) | **0.7746** (worst) |
| vision, pooled | 0.8216 | 0.8152 |
| telemetry only | **0.8530** (highest) | **0.8419** (best) |

The better an arm fits the training data, the worse it generalizes — a textbook overfitting
signature rather than a sign that vision is merely uninformative. An uninformative-but-
harmless feature would leave both columns roughly unchanged. Note also where pooling lands:
averaging ~4.5 frames per token *raises* training loss relative to a single frame, because
the average is smoother and carries less memorizable per-frame idiosyncrasy — and its
validation score improves correspondingly.

**2. Pooling recovers most of the gap, which answers the question arm 3 was built to ask.**
Giving vision the same aggregate-don't-discard treatment telemetry already gets takes the
deficit from -0.067 to -0.027. So a substantial part of "vision hurt" was an artifact of
handing each token a single still photograph while the motion branch summarised its entire
150 ms window. It was not the whole story — -0.027 remains — but it was most of it.

**3. The reported delta depends on an arbitrary stopping point.** We report the final epoch.
The vision arm peaked at 0.8096 on epoch 25 and fell to 0.7746 by epoch 30, and its curve is
visibly noisier throughout (0.700 → 0.760 → 0.736 → 0.799 → 0.810 → 0.775) than the control's
near-monotone climb. So:

- vision nearest: **-0.0673** at the final epoch, **-0.0323** at each arm's best epoch
- vision pooled: **-0.0266** either way (it peaked at epoch 30)

Both negative under both conventions, so the direction is stable. The magnitude of the
nearest arm's deficit is not — it roughly halves. And note that **both the control and the
pooled arm were still improving at epoch 30**, so the 30-epoch budget is truncating them;
the pooled arm in particular climbed 0.8005 → 0.8152 over the final five epochs.

**There is no early stopping or best-epoch selection for the classifier.** `run_training`
reports whatever epoch 30 happened to produce. All three arms are treated identically so the
comparison is fair, but the numbers are more fragile than a single figure suggests, and two
of the three arms were cut off mid-climb. This is the same mistake found and fixed in the
tokenizer, still present in the classifier.

---

## Why this run is more trustworthy than any before it

Five earlier attempts produced no usable number. Each failure bought a fix, and five of those
fixes bear directly on whether this result means anything:

| fix | why it mattered to this number |
|---|---|
| **Fusion branch normalization** | Vision entered the fusion layer ~11-26x louder than motion, which biased the comparison *toward* vision helping or hurting unpredictably. Now 3.46x, per-element parity. Without this, "vision hurt" could just have been "the fusion layer was miswired." |
| **DINOv2 preprocessing** | Frames were squashed to 224x224 with no ImageNet normalization. Degraded features bias directly toward "vision doesn't help" — exactly the result we measured, so this had to land first. |
| **Corpus-level F1@50** | The original macro-averaged per-demo, which is not the protocol Nomadic/M2R2 report. |
| **T-MSE smoothing loss** | Cross-entropy alone has no objection to a prediction flipping class between neighbouring tokens; F1@50 punishes it heavily. Both arms now train on MS-TCN's actual objective. |
| **Tokenizer gates on held-out demos** | A collapsed codebook (measured: 8 of 512 effective codes) would have starved *both* arms. This run's tokenizer was verified healthy before a single classifier epoch ran. |

Plus the protocol hygiene that makes it a measurement rather than an anecdote: a real
validation split with `test_split1` untouched, per-run directories holding config,
checkpoint, logs and results, and per-arm persistence.

---

## What could still change the answer

Ordered by how much they could move it.

1. **Reproducibility is unmeasured.** One run. The delta is -0.067 and we cannot yet say
   whether run-to-run variance is ±0.01 or ±0.10. Until repeats exist this number has no
   error bar. Seeding has landed; the study has not run.
2. **No classifier early stopping.** Halves the delta depending on where you stop (above).
3. **Vision is temporally subsampled, not aggregated.** Each token gets one still photo, so
   the vision branch structurally cannot contribute motion information — only scene context.
   Arm 3 (pooled over the window) tests exactly this and has not run.
4. **The frame-invariant component of the DINOv2 feature.** ~87% of the CLS token's magnitude
   is constant across frames; LayerNorm rescales but does not remove it. Per-dimension
   standardization over training frames is the next escalation.
5. **Codebook initialization.** `uniform_(-1/num_codes, 1/num_codes)` starts all 512 codes
   within ±0.002 of the origin while encoder outputs sit at radius ~0.59 — measured, only 2
   of 512 codes win at init. Roughly the first 10 epochs are spent undoing this rather than
   learning motion. Affects both arms equally, but caps the representation both depend on.
6. **Boundary precision 0.115.** The tokenizer changes token far more often than actions
   change. Both arms inherit this ceiling.
7. **Validation is not test**, and the splits are not distributionally identical (`Align`
   13.40% train vs 18.35% val; `Pull` 5.79% train vs 7.88% test).

## Why vision plausibly hurt

Not a settled explanation; the candidates, in the order the evidence supports them:

- **Capacity without information.** Best supported: the vision arm fits training data better
  and generalizes worse. 384 extra dimensions against ~199k training tokens and 10 classes.
- **Static context is the wrong signal for segmenting motion.** One frame per token can say
  *what is in view*, never *what is moving*. Action boundaries are dynamic events. Telemetry
  already carries the dynamics, so vision may be adding scene identity the classifier does
  not need and can overfit to.
- **Camera is static and the scene changes slowly**, so adjacent tokens' vision features are
  near-duplicates, adding correlated noise rather than discriminative signal.

**Arm 3 has now discriminated between the first two, and the answer is "both".** Pooling
recovered 60% of the deficit, so subsampling was real and substantial. The residual -0.027
survives proper aggregation, so capacity-without-information or plain irrelevance accounts
for the rest — consistent with M2R2 measuring vision-only at 21.6 F1@50 on this dataset.

---

## Superseded runs

| run | outcome |
|---|---|
| Sept 9, two runs | Predate per-run directories. First: arm 1 trained, score lost (printed only at the end). Second: killed to fix logging. |
| `20260910-145157` | Stalled at tokenizer epoch 17/20, marked SYSTEM_FAILURE. Later attributed to the laptop running on battery with the GPU clamped to 180 MHz / 15 W. |
| `20260911-104521` | Killed at epoch 11, cause unknown, stderr discarded. Checkpoints survived. |
| `20260911-113845` | Gate FAILED (8 effective codes) after resuming from a "best loss" checkpoint. Cost one minute and revealed that VQ-VAE loss anti-correlates with codebook health. |

**A note on every wall-clock number produced before ~13:22 on 2026-09-11:** the laptop was on
battery and the GPU was clamped to 180 MHz / 15 W. Identical work measured 601.7 s throttled
and 24.0 s on AC — 25x. Classifier epochs: ~17 s throttled, 0.8 s on AC. Accuracy numbers are
unaffected; timings from that period are not comparable to anything. Run configs now record
`pstate`, `clocks.sm` and the enforced power limit alongside every measurement.


---

## Files in this folder

| path | what it is |
|---|---|
| `images/f1_comparison.png` | F1@50 per arm with the delta against the control |
| `images/classifier_training.png` | per-arm training loss and validation F1@50 during training |
| `images/tokenizer_training.png` | tokenizer loss and codebook utilization per epoch, gate threshold marked |
| `data/config.json` | git commit, torch version, command line, every hyperparameter, full demo membership of all three splits |
| `data/results.json` | each arm's F1@50, wall time, and complete per-epoch history |
| `data/tokenizer_report.json` | the three gate metrics on held-out demos |
| `data/tokenizer_history.json` | per-epoch tokenizer train/held-out loss and codebook utilization |
| `data/run.log` | full timestamped run log |

Regenerate the figures from any run directory with:

```bash
python -m temporal_classifier.plots runs/<run-id> --out-dir results/<version>/images
```
