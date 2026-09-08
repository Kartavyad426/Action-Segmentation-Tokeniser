# robotics — action segmentation & labeling pipeline

Unrelated to the tau2-bench harness-evolution project one level up (`../claude.md`).
This is a separate effort: building a labeled training dataset (and eventually a model) for
segmenting robot demonstration recordings into small action chunks with structured labels.

## What this is

Offline pipeline: video + robot telemetry (joint angles, gripper state, force/torque) for
~100-200 recorded demos → temporal chunks, each tagged with a closed-vocabulary action label
(e.g. reach, grasp, lift, place, insert...), verified by a human reviewer, exported as
training data.

**Status: design/brainstorming phase as of 2026-09-08. No code written yet.** Read
`HANDOFF.md` first — it has the current design, the reasoning behind each decision, and
what's still open.

## Scope

- Two-phase pipeline:
  1. **Bootstrap** (~first 20-30 demos): telemetry change-point boundary detection + VLM
     labeling from a seed-and-grow closed vocabulary + human review.
  2. **Scale-up** (remaining demos): a trained auto-labeler (motion tokenization + vision
     enrichment + temporal classifier) replaces per-demo VLM calls, still human-reviewed.
- The vocabulary is a closed set of manipulation primitives, grown iteratively during Phase 1
  rather than hand-authored upfront — tasks/objects are unseen in advance, but primitives are
  expected to stay stable across them.
- Telemetry is available at the eventual deployed model's inference time too, not just during
  labeling — so every exported chunk carries telemetry, not just video + label.

## What this is not (yet)

- **Not a deployed control policy.** Everything here is about building labeled training data.
  A downstream-policy architecture question came up mid-brainstorm and was explicitly
  descoped back into this pipeline once it became clear it was actually about the Phase 2
  labeling model, not a control policy. If a real deployed-policy design happens later, it's
  its own sub-project with its own brainstorming pass.
- **Not implemented.** No pipeline code, no model training, no data collected in this repo yet.

## Directory structure (planned, not yet built)

```
robotics/
├── CLAUDE.md             # this file
├── HANDOFF.md            # current design + open questions — read this first
├── boundaries/           # Phase 1: telemetry change-point detection
├── labeling/             # Phase 1: VLM labeling + seed-and-grow vocabulary
├── review/                # human review integration (Label Studio)
├── tokenizer/             # Phase 2: unsupervised motion tokenization (VQ-VAE)
├── temporal_classifier/   # Phase 2: sequence model over enriched motion tokens
└── data/                  # video + telemetry demos (do not commit raw data — gitignore it)
```

## Conventions

- Follow the repo-wide git/testing conventions from `../claude.md` unless this file says
  otherwise.
- This workspace will likely need its own dependencies (vision/robotics libs). Keep them out
  of any root-level dependency file if the two projects' environments diverge — don't couple
  this project's deps to the harness-evolution project's.
