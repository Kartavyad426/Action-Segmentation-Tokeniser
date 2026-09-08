# Robotics: Action Segmentation Pipeline — Handoff

Last updated: 2026-09-08. **Status: design/brainstorming phase, no code written yet.**
This doc is the entry point for whoever (agent or human) picks this up next.

## Status

- Branch: none yet — this folder is new, created from `loop-breaker-repeat-rule` at the repo
  root (that branch is unrelated work; don't assume it applies here).
- Commits: none.
- Tests: none.
- This document captures the output of a brainstorming session
  (`superpowers:brainstorming`, architectural path). **No implementation has started, and the
  design below has not gone through a formal spec self-review or user sign-off on a written
  spec.** If picking this up as Claude Code, the next step is finishing that process: write
  `docs/superpowers/specs/YYYY-MM-DD-robotics-action-segmentation-design.md` at the repo
  root, self-review it, get the user's explicit approval, only then invoke
  `superpowers:writing-plans`. Do not start implementing straight from this handoff doc.

## What this project is

Build a labeled training dataset from ~100-200 robot manipulation demos: video + synced robot
telemetry (joint angles, gripper state, force/torque), offline, post-hoc. Goal: segment each
demo into small temporal chunks, each tagged with a structured action label from a closed
vocabulary (not free-form captions), with a human-in-the-loop review step. Telemetry is
available at the eventual deployed model's inference time too, not only during labeling.

## Key decisions made (and why)

1. **Closed vocabulary, but seeded-and-grown, not hand-authored upfront.** Tasks/objects are
   unseen in advance, but manipulation primitives (reach, grasp, lift, place, insert,
   rotate...) are expected to stay stable across unseen tasks. Start from a small seed
   vocabulary, let the labeling model propose a new label when nothing fits, human reviewers
   approve/merge into the vocabulary, freeze once new-label proposals taper off.

2. **Two-phase pipeline, not one model from the start.**
   - **Phase 1 (bootstrap, ~first 20-30 demos):** telemetry-based change-point detection for
     segment boundaries (gripper-state transitions, velocity zero-crossings, force/torque
     spikes — classical signal processing, e.g. `ruptures`/PELT, no training needed) → VLM
     assigns a label from the current vocabulary per candidate segment → human reviews/corrects
     both boundaries and labels. Label Studio's video-timeline UI is the closest fit found for
     the review step; CVAT is more frame/bbox-oriented and less suited to segment review.
   - **Phase 2 (scale to remaining ~150-180 demos):** replace the per-demo VLM calls with a
     trained auto-labeler once Phase 1 has produced enough seed labels (architecture below).
     Still human-reviewed — cheaper and more accurate per demo, not zero-review.

3. **Phase 2 architecture: motion tokenization + vision enrichment + temporal classifier** —
   not M2R2's late fusion, and not full early/joint fusion of raw per-modality tokens either:
   - Unsupervised motion tokenizer (VQ-VAE-style) turns raw telemetry into a discrete sequence
     of "motion tokens" — trained on reconstruction only, no labels needed. Telemetry is
     high-frequency, so this step shouldn't be data-starved even at our demo count.
   - Each motion token is enriched with a time-aligned visual feature (frozen pretrained
     encoder, e.g. CLIP/DINOv2) via concatenate+project or additive fusion — vision as an
     enrichment on the motion-token spine, not a symmetric second stream. Cross-attention
     (motion tokens attending over nearby visual tokens) is a fallback upgrade if
     object-identity cases need more than simple additive fusion provides.
   - A temporal classifier (transformer or TCN, e.g. MSTCN-style) reads the full enriched
     token sequence with before/after context — this is the single biggest lever per the
     research below: raw tokens through a temporal classifier tripled F1@50 over
     context-free classification, before vision was even added.
   - **Before committing:** validate a telemetry-only baseline first and measure the actual
     lift from adding vision. M2R2's own ablation found vision nearly useless in *their*
     fusion design (74.5% → 74.6% F1@50) — don't assume Nomadic's larger lift (79.5% → 93.1%)
     transfers to our data without checking.

4. **Every exported chunk carries telemetry, not just video/label** — the eventual deployed
   model will also have telemetry at inference time, so the dataset shouldn't need
   re-deriving later if/when a downstream policy wants video+telemetry fusion.

5. **A separate "downstream deployed policy" design question was raised mid-session and
   explicitly descoped back into this pipeline** once we established the joint vision+
   telemetry transformer being discussed was actually about the Phase 2 labeling model, not a
   control policy. If a real deployed-policy design conversation happens later, treat it as
   its own sub-project with its own brainstorming pass — don't assume anything decided here
   (architecture, fusion style) about action representation, control frequency, or policy
   design for that separate effort.

## Research grounding (read before re-deriving any of this)

- **M2R2** (arXiv 2504.18662) — modular late-fusion temporal action segmentation (TAS):
  per-modality encoders (ActionCLIP vision, AST audio, learned proprioception projection)
  pooled per-window, one shallow self-attention fusion layer, fed into standard TAS heads
  (MSTCN/ASRF/DiffAct). Ablation: proprioception alone 74.5% F1@50, adding vision/audio only
  reaches 74.6% — vision barely helped *in their specific fusion design*. Results: REASSEMBLE
  82.4%, PerfectPour 86.0%, JIGSAWS 89.4% F1@50.
- **Nomadic AI blog**, "Segmenting the Key Micro-Actions in Robotic Footage"
  (nomadicai.com/blog/action-segmentation-benchmark) — unsupervised motion tokenization +
  temporal classifier (full before/after context) + optional camera fusion *into* motion
  tokens. On REASSEMBLE (~148 demos, close to our scale): 93.1% F1@50 vs. M2R2's 83.4%.
  Context alone (before vision was added) tripled a random-forest baseline from 25.8% to
  79.5%. **Caveat: vendor blog post, no linked paper or repo found — numbers are not
  independently verified the way M2R2's arXiv result is.** Treat as a promising direction to
  prototype, not a settled benchmark.
- Other named prior art on telemetry-driven segmentation: **AWE** (waypoint selection from
  proprioceptive deviation), **DexSkills** (haptic/force-torque segmentation), **LOTUS** /
  **BUDS** (vision-feature-clustering-based — less relevant here since our design leans
  telemetry-first).

## What remains / next steps

1. Write the formal spec (`docs/superpowers/specs/YYYY-MM-DD-robotics-action-segmentation-design.md`
   at the repo root, per the `superpowers:brainstorming` skill's convention) before any code —
   this handoff doc is a summary, not a substitute for that self-reviewed, user-approved spec.
2. Confirm the actual telemetry schema for the real robot (exact fields, sample rate). The
   design above assumes joint angles/pose, gripper state, force/torque at roughly 20 Hz, based
   on precedent (REASSEMBLE's ~20 Hz), not yet confirmed against the real setup.
3. Decide/prototype the seed vocabulary (which primitives to start with) — needed before
   Phase 1 can run at all.
4. Prototype the telemetry-only Phase 2 baseline and measure vision's actual lift before
   building the full motion-tokenizer + vision-enrichment + temporal-classifier stack.
5. Set up Label Studio (or confirm an alternative) for the human review step.
6. **Open, not yet decided:** does Phase 2's temporal classifier also relearn segment
   boundaries (like M2R2/Nomadic do), or keep reusing Phase 1's heuristic boundaries and only
   classify within them? Current lean is the latter (simpler, less data-hungry) but this
   wasn't settled in discussion.

## Gotchas

- Don't conflate this project with the tau2-bench harness-evolution project one level up
  (`../claude.md`, `../docs/RESUME.md`) — different goal, different domain, no shared code or
  conventions beyond general repo hygiene.
- 100-200 demos is small for training anything from scratch. Every design choice above was
  made with that constraint in mind (frozen pretrained vision encoders, unsupervised
  tokenizer, reusing Phase 1 boundaries in Phase 2 instead of relearning them). Don't casually
  add a component that needs more supervised data than that without checking scale first.
