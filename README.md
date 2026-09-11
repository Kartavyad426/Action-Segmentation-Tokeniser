# Action Segmentation with a Motion Tokeniser

Temporal action segmentation of robot manipulation demonstrations, built to answer one
question by ablation:

> **Does adding vision improve temporal action segmentation over telemetry alone?**

Robot telemetry is compressed into discrete motion tokens by a VQ-VAE, each token is
optionally enriched with a frozen DINOv2 feature from the time-aligned video frame, and an
MS-TCN predicts a per-token action label. The same pipeline runs with and without the vision
pathway, and the difference in segmental F1@50 is the result.

Validated on [REASSEMBLE](https://arxiv.org/pdf/2502.05086), 148 demonstrations of
contact-rich assembly and disassembly.

## Current result

| arm | F1@50 | vs control |
|---|---|---|
| telemetry only *(control)* | **0.8419** | — |
| vision, 1 frame per token | **0.7746** | **-0.067** |
| vision, pooled over token window | **0.8152** | **-0.027** |

**Vision did not help in either sampling regime**, though pooling every frame in a token's
window rather than taking one recovers about 60% of the deficit — so temporal subsampling was
a substantial part of why the naive version hurt. Directionally consistent with
[M2R2](https://arxiv.org/html/2504.18662), whose ablation on this same dataset found vision
adding +0.1 to proprioception (74.5 → 74.6) and vision alone scoring 21.6.

**These are validation numbers from a single run.** The official test split is untouched, and
no reproducibility study has run yet, so nothing here is comparable to published work.
See [`docs/are-our-results-good.md`](docs/are-our-results-good.md) for the full assessment,
including why the result is believable, why it should not yet be believed, and what would
settle it.

Full write-up, figures and raw artifacts: [`results/v1/`](results/v1/README.md).

## Pipeline

```
demo.h5 ──┬─ telemetry 5 channels @ ~1000 Hz ─ resample 100 Hz ─ 150 ms windows ─┐
          │                                                                      │
          │                                             VQ-VAE ─ 32-d motion token
          │                                                                      │
          ├─ hama1 video @ 30 Hz ─ frame nearest each token ─ frozen DINOv2 ─ 384-d
          │                                                                      │
          │                                              LayerNorm both ─ concat ─ project
          │                                                                      │
          └─ low-level segment labels ─ per-token targets ───────── MS-TCN ─ F1@50
```

Details and measured shapes at every hop: [`docs/pipeline-dataflow.md`](docs/pipeline-dataflow.md).

## Running it

```bash
pip install -r requirements.txt
python data/download_reassemble.py --workers 10      # ~250 GB

# tuning runs: train on 89 demos, score on 22 validation demos
python -m temporal_classifier.compare --eval-on val

# figures from any run directory
python -m temporal_classifier.plots runs/latest

# aggregate F1@50 across repeated runs, with the paired delta
python -m temporal_classifier.summarize runs/
```

Must be invoked as `python -m temporal_classifier.compare`, not by file path.

Each run creates `runs/<timestamp>/` holding its config (git commit, torch version, every
hyperparameter, full demo membership of all three splits, GPU power state), its tokenizer
checkpoint, per-epoch history, gate report, and per-arm results written the moment each arm
completes. Runs are never overwritten. See
[`docs/design.md`](docs/design.md#how-to-run) for flags including `--resume-from`.

## Repository

| path | |
|---|---|
| `tokenizer/` | VQ-VAE motion tokenizer, plus its three label-free quality gates |
| `enrichment/` | frozen DINOv2 features and the fusion layer |
| `temporal_classifier/` | MS-TCN, F1@50 metrics, the three-arm comparison, plots |
| `results/` | versioned results: write-up, figures, raw artifacts, one folder per usable run |
| `docs/` | design, data flow, hyperparameters, issues and fixes, results assessment |
| `tests/` | 140 tests |

## Notable design decisions

Several exist because a first implementation got them wrong in ways that would have biased
the very measurement this project makes. Each is documented with the evidence in
[`docs/issues-and-fixes.md`](docs/issues-and-fixes.md).

- **The fusion layer LayerNorms each branch separately, before concatenation.** Unnormalized,
  the DINOv2 feature entered ~11–26x louder than the motion token, and because `nn.Linear`
  cannot know which columns belong to which modality, that scale gap *is* the
  branch-dominance gap. Normalizing the concatenated vector instead does not work: one shared
  scalar over both halves leaves the ratio unchanged.
- **Tokenizer checkpoints are selected by codebook utilization, not loss.** Total VQ-VAE loss
  *falls* when the codebook collapses, because the commitment and codebook terms shrink toward
  zero. The lowest-loss checkpoint we measured had 8 live codes of 512 and failed the gate.
- **Boundary alignment is scored against its own chance level.** Pure recall gives a perfect
  1.0 to a tokenizer that changes token every window, which is what the original metric did.
- **Classifier training includes MS-TCN's truncated-MSE smoothing loss.** Cross-entropy has no
  objection to a prediction flipping class between neighbouring tokens; F1@50 punishes it
  heavily.
- **Validation is carved from the official train split**, and the assignment is derived from
  the canonical split file rather than from what is on disk, so membership cannot shift as a
  dataset finishes downloading.

## Status

Research code under active development. The vision ablation has a first answer; the
reproducibility study and the held-out test evaluation have not run.
