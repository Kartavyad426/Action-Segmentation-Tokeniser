# Are our results good?

Two questions, kept separate because they have different answers:

1. **Why is vision hurting?** — we have evidence, and the literature broadly predicts it.
2. **Is our telemetry-only number actually better than published work?** — unknown, and the
   honest answer is that we cannot say yet.

Numbers referenced here are from `results/v1` (validation split, single run, no error bar).

---

## 1. Why vision is hurting

Measured: telemetry-only F1@50 **0.8419**, vision at one frame per token **0.7746**, delta
**-0.067**. Candidate explanations, ranked by how well *our own* evidence supports them.

### Capacity without information — best supported

| | final train loss | val F1@50 |
|---|---|---|
| telemetry only | 0.8530 | **0.8419** |
| vision, 1 frame/token | **0.7913** | 0.7746 |

The vision arm fits the training data **better** and generalizes **worse**. That distinction
matters: a feature that was merely uninformative would leave both curves roughly unchanged.
Ours traded generalization for fit. 384 extra input dimensions against ~199k training tokens
and 10 classes buys capacity to memorize.

### One frame per token carries no motion

Arm 2's vision feature is a single instantaneous snapshot; the other ~3.5 frames inside the
token's 150 ms window are discarded. It can express *what is in view*, never *what is
moving*. Action boundaries are dynamic events, and telemetry already carries the dynamics —
so vision contributes scene identity the classifier does not need and can latch onto.

Note the asymmetry this creates with the motion branch: a motion token *aggregates* all 15
telemetry samples in its window, while the vision feature *subsamples* to one frame out of
~4.5. Arm 3 (mean-pooling every frame in the window) exists to test exactly this.

### Low information density inside the feature

Measured on real DINOv2 features: ~87% of the CLS token's magnitude is a component constant
across frames (norm 43.49 of a 46.54 total, per-frame residual 16.26). The informative
per-frame signal is a small residual on a large constant, so most of those 384 dimensions
carry little discriminative variance — while all of them carry parameters.

### Static camera, slowly-changing scene

Adjacent tokens' vision features are near-duplicates, adding correlated noise rather than
independent evidence.

---

## 2. What the literature says

### The most relevant prior result is an ablation on *this exact dataset*

[M2R2](https://arxiv.org/html/2504.18662) (arXiv 2504.18662), Table IV, REASSEMBLE, F1@50:

| modality | F1@50 |
|---|---|
| vision only | **21.6** |
| audio only | 36.4 |
| proprioception only | **74.5** |
| all modalities | **74.6** |

**Vision alone is nearly useless on REASSEMBLE, and adding it to proprioception buys +0.1.**
Their explanation is dataset-specific: the objects are small, frequently occluded, and
visually near-identical — they single out distinguishing pegs that differ by millimetres. The
[REASSEMBLE dataset paper](https://arxiv.org/pdf/2502.05086) says the same, describing 149
recordings with "strong visual similarity between objects". A vision-only BR-Prompt feature
baseline scores **5.7** F1@50.

So the honest framing of our finding is not "we discovered vision does not help." It is:
**vision was never going to help much here, and published work said so.** What we add is that
in a concat+project design it does not merely fail to help — it actively costs 6.7 points.

### Why ours hurts where M2R2's was neutral

Two concrete architectural differences, both plausible contributors:

- **Fusion mechanism.** M2R2 fuses with a modality transformer (a self-attention encoder
  layer followed by an MLP) — a late-fusion design where attention can learn to attenuate a
  useless modality toward zero. We use concat → linear → MS-TCN: the vision dimensions are
  present in every input vector and cannot be gated away. A late-fusion design has an easy
  path to "ignore this"; ours does not.
- **Vision encoder.** M2R2 uses ActionCLIP (video/action-pretrained, 512-dim). We use DINOv2
  (image-level, self-supervised, 384-dim). Work on frozen features for temporal tasks finds
  DINOv2's spatial context [lacks motion information and struggles to fully represent the
  action](https://arxiv.org/pdf/2309.05590), with video-pretrained features such as
  VideoMAEv2 outperforming it on action segments.

### The general failure mode is well documented

[What Makes Training Multi-modal Classification Networks Hard? (Wang et al., CVPR
2020)](https://ai.meta.com/research/publications/what-makes-training-multi-modal-classification-networks-hard/)
shows multimodal networks can underperform their best *unimodal* counterpart regardless of
fusion mechanism or regularization, because **different modalities overfit and generalize at
different rates**, so training them jointly under one optimization strategy is sub-optimal.
Their remedy (Gradient-Blending) weights each modality's gradient by its overfitting ratio.

Related framings: [modality competition, where only dominant modalities are fully explored by
joint training](https://arxiv.org/pdf/2507.10203), and [greedy learning addressed by
on-the-fly gradient modulation](https://arxiv.org/html/2405.07930v1).

That is a close description of our two curves.

### What this suggests trying, if we want vision to have a fair hearing

In rough order of cost:

1. **Arm 3** (already running) — pooling frames tests whether subsampling was the problem.
2. **A gated or attention-based fusion** rather than concat+project, giving the model a path
   to attenuate vision. This is the M2R2 design difference most likely to explain the gap.
3. **Per-dimension standardization** of the DINOv2 feature over training frames, removing the
   frame-invariant component that LayerNorm only rescales.
4. **A video-pretrained encoder** (VideoMAEv2, ActionCLIP) instead of an image-level one.
5. **Gradient blending or modality-balanced optimization**, the literature's direct remedy for
   the overfitting-rate mismatch we appear to be seeing.

Note that 2-5 all make vision *more* likely to help. If the goal is an honest measurement of
"does vision help in a simple concat+project design", the current result already answers that
and these are a different question.

---

## 3. Is our telemetry-only number better than published work?

| | F1@50 | measured on |
|---|---|---|
| **ours, telemetry only** | **84.19** | validation (22 demos, carved from the train half) |
| M2R2, proprioception only | 74.5 | REASSEMBLE official test split |
| M2R2, best configuration (with ASRF) | 82.4 | REASSEMBLE official test split |
| Nomadic, "context alone" | 79.5 (claimed) | unspecified |

**Unknown. Do not quote this as beating anything.** Three reasons it may not hold:

1. **Validation is not test.** Ours is 22 demos carved from the official *train* half; theirs
   is the 37-demo official test split. We measured that the splits are not distributionally
   identical — `Pull` is 3.30% of validation but 7.88% of test, `Align` 18.35% vs 13.41%.
2. **One run, no error bar.** We cannot yet distinguish a real 10-point margin from seed
   variance. The reproducibility study is what settles this.
3. **Protocol details move F1@50 substantially** — segment construction, background handling,
   and corpus-level versus macro-averaged aggregation.

### One inflation route checked and cleared

Consecutive segments sharing a label merge into a single segment when labels are projected
onto the token grid, which would make IoU matching artificially easy. Measured over 10
validation demos:

```
ground-truth low-level segments in the files : 668
segments the F1 metric actually scores       : 654   (2% lost to merging)
median per demo: 62 -> 61
```

2% is not a meaningful inflation. This particular worry is dead.

### Reasons the number might be genuine

We differ from M2R2 in ways that could legitimately help the proprioception path: a VQ-VAE
motion tokenizer instead of a non-discretized learned projection, MS-TCN's T-MSE smoothing
loss, and corpus-level F1 aggregation. If 84 survives on the test split, then **"a discrete
motion tokenizer beats a learned continuous proprioception encoder" is a more interesting
result than anything we have found about vision.**

### On the Nomadic numbers

Their blog is JavaScript-rendered and returns only a page title to a fetcher, so the
79.5 → 93.1 figures remain unverified — consistent with what
`docs/differences-from-published-results.md` already records: a vendor blog post with no
linked paper or repository. Their claimed 93.1 also sits well above M2R2's best *published*
82.4 on the same dataset, which warrants scepticism given M2R2's own vision ablation shows
vision contributing +0.1.

### What would settle it

1. Run the reproducibility study, so the telemetry-only number arrives with a spread and we
   know what zero looks like.
2. Freeze hyperparameters, then run `--eval-on test` **once**.

Until both are done, every number in `results/` is a validation number and is not comparable
to anything published.

---

## Sources

- [M2R2: MultiModal Robotic Representation for Temporal Action Segmentation](https://arxiv.org/html/2504.18662)
- [REASSEMBLE: A Multimodal Dataset for Contact-rich Robotic Assembly and Disassembly](https://arxiv.org/pdf/2502.05086)
- [What Makes Training Multi-modal Classification Networks Hard? (Wang et al., CVPR 2020)](https://ai.meta.com/research/publications/what-makes-training-multi-modal-classification-networks-hard/)
- [Improving Multimodal Learning via Imbalanced Learning](https://arxiv.org/pdf/2507.10203)
- [Improving Multimodal Learning with Multi-Loss Gradient Modulation](https://arxiv.org/html/2405.07930v1)
- [MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://arxiv.org/abs/1903.01945)
- [Temporal Action Localization with Enhanced Instant Discriminability](https://arxiv.org/pdf/2309.05590)
