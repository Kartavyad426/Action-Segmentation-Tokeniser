# Phase 2 Core: Motion Tokenizer + Vision Enrichment + Temporal Classifier — Design

Status: draft, pending user review.
Author: brainstorming session, 2026-09-08 (Claude Code + user).
Parent context: `HANDOFF.md` (full two-phase pipeline design). This spec covers **one
sub-project** carved out of that larger design — see Scope below.

## Background

`HANDOFF.md` documents a two-phase pipeline for turning ~100-200 robot demo recordings
(video + telemetry) into labeled training data: Phase 1 bootstraps a closed action
vocabulary via telemetry change-point detection + VLM labeling + human review; Phase 2
replaces per-demo VLM calls with a trained auto-labeler (motion tokenizer + vision
enrichment + temporal classifier), modeled on the approach described in Nomadic AI's blog
post "Segmenting the Key Micro-Actions in Robotic Footage."

The user's real demo data location/schema is not yet confirmed (someone needs to be asked
internally). Rather than block on that, this spec scopes down to building and validating
Phase 2's core architecture against **REASSEMBLE**, the open dataset Nomadic's own reported
numbers (93.1% F1@50) and M2R2's (83.4% F1@50) are benchmarked on, and which ships its own
ground-truth segment boundaries and labels — meaning it needs no Phase 1 bootstrap at all.

## Goals

- Build a working motion tokenizer → vision-enrichment → temporal-classifier stack in
  PyTorch, in a dedicated virtualenv for this project.
- Validate it end-to-end on REASSEMBLE, producing an F1@50 number comparable to the
  reference numbers already in `HANDOFF.md`.
- Answer the open question `HANDOFF.md` explicitly flagged: does vision enrichment give a
  real lift over a telemetry-only baseline in *our* implementation, the way Nomadic reports
  (79.5% → 93.1%) and unlike M2R2's own ablation (74.5% → 74.6%)? This is a go/no-go check
  on keeping the vision-enrichment component at all.
- Leave the tokenizer open to pretraining on additional open manipulation datasets beyond
  REASSEMBLE (it needs no labels, so this is cheap) if that improves reconstruction quality
  or codebook utilization — not required for v1, but the code should not assume
  REASSEMBLE-only.

## Non-goals (explicitly out of scope for this spec)

- Phase 1 (telemetry change-point boundary detection, VLM labeling, seed-and-grow
  vocabulary, human review loop). Blocked on confirming the real robot's telemetry schema
  and getting access to real demo data — separate spec, later.
- Label Studio / any human review tooling integration — separate spec, later.
- Any work on the user's real demo data — this spec only touches REASSEMBLE (+ optionally
  other open datasets for tokenizer pretraining).
- A deployed control policy of any kind — out of scope for this entire project per
  `HANDOFF.md`.
- Relearning segment boundaries in Phase 2 (the open question in `HANDOFF.md` item 6). Not
  relevant here since REASSEMBLE already provides ground-truth boundaries; this spec's
  classifier is evaluated against those, not against boundaries Phase 2 invents itself.

## Architecture

Three components, each independently testable:

### 1. Motion tokenizer (`tokenizer/`)

VQ-VAE-style: an encoder compresses short windows (~100-200ms, exact value tunable) of
raw telemetry into a continuous latent vector; a vector-quantization bottleneck snaps that
vector to the nearest entry in a learned codebook (starting size ~512 entries, tunable); a
decoder reconstructs the original window from the quantized vector. Trains on
reconstruction loss only — no labels needed, so every timestep of every demo (and
potentially other open datasets) is usable training signal despite the small demo count.

Output: each demo's telemetry stream becomes a sequence of discrete token IDs.

### 2. Vision enrichment (`enrichment/`)

Attaches a time-aligned visual feature to each motion token, pulled from a frozen
pretrained vision encoder (DINOv2 or CLIP, small/base variant given the 8GB GPU budget) run
on the corresponding video frame. Fusion mechanism: concatenate + project, or additive
fusion — matching `HANDOFF.md`'s existing decision. Cross-attention fusion (motion tokens
attending over nearby visual tokens) is an explicit fallback upgrade, built only if
concat/additive fusion proves insufficient once measured — not built in v1.

### 3. Temporal classifier (`temporal_classifier/`)

Reads the full enriched token sequence for a demo, with complete before/after context, and
predicts a per-token action label. **Architecture choice: MSTCN-style TCN (multi-stage
temporal convolutional network, stacked dilated convolutions for a large receptive field),
not a Transformer, to start.** Rationale: proven at this exact data scale (M2R2 uses MSTCN
heads successfully at 100-150 demo counts), cheaper to train on an 8GB GPU, and it follows
the same staged pattern already applied to vision fusion — start with the simpler,
cheaper option, escalate to Transformer only if the TCN can't capture long-range context
well enough once measured.

This is the only supervised component — trained on REASSEMBLE's ground-truth per-frame
action labels. Full sequence context matters enormously here: per the research cited in
`HANDOFF.md`, feeding full before/after context through a sequence model (vs. context-free,
per-token classification) tripled F1@50 on its own, before vision was even added — which is
why this cannot be a plain MLP; it must be a model that aggregates information across
neighboring timesteps.

## Data flow & milestones

1. **Acquire + inspect REASSEMBLE.** Confirm actual telemetry schema and sample rate
   against the assumption inherited from `HANDOFF.md` (~20Hz, joint angles/pose, gripper
   state, force/torque) — this has not been verified against the real dataset yet. Confirm
   demo count and typical demo duration (needed to size training-time expectations, which
   are currently rough estimates — see Estimates below).
2. **Train the tokenizer** on REASSEMBLE telemetry (optionally mixing in another open
   dataset). Run the three tokenizer-level tests below.
3. **Train the telemetry-only classifier** (tokenizer output, no vision) → baseline F1@50.
4. **Add vision enrichment, retrain the classifier** → measure the actual lift vs. the
   baseline. This answers the go/no-go question in Goals.
5. **Report:** F1@50 for (a) telemetry-only and (b) vision-enriched, next to the reference
   numbers already in `HANDOFF.md` (Nomadic 93.1%, M2R2 83.4%).

## Testing

Four checks, split by what they can validate without labels vs. what needs them:

**Tokenizer-only (no labels required):**
1. **Held-out reconstruction error** — encode→quantize→decode telemetry windows the
   tokenizer never trained on; measure per-channel normalized reconstruction error. Confirms
   the codebook captures the real motion distribution rather than memorizing training demos.
2. **Codebook utilization** — usage histogram/entropy over held-out data. Catches codebook
   collapse (a small subset of codes absorbing all the traffic, leaving the rest dead),
   which reconstructs fine but starves the classifier of discriminative signal — a failure
   mode invisible to check 1 alone.
3. **Segment-boundary alignment probe** — using REASSEMBLE's ground-truth boundaries
   (without training anything supervised): do token *changes* cluster near ground-truth
   action transitions more than chance, and is the token sequence relatively stable within a
   single ground-truth segment? Cheap sanity check that the tokenization is carving motion
   at task-relevant joints before investing in the full classifier.

**Classifier-level (needs labels):**
4. **F1@50 on REASSEMBLE's test split**, computed twice — telemetry-only and
   vision-enriched — against the reference numbers in `HANDOFF.md`.

Checks 1-3 belong to the tokenizer component's own test suite and gate moving on to the
classifier. Check 4 is the milestone-level success criterion for the whole sub-project.

## Environment

- Python, PyTorch. Dedicated virtualenv for this project — not shared with the tau2-bench
  harness-evolution project (`../tau2-bench`) or any other project in this workspace, per
  `CLAUDE.md`'s instruction to keep dependencies decoupled.
- GPU: NVIDIA RTX PRO 1000 Blackwell (laptop), 8GB VRAM. Sized model choices (small/base
  vision encoder variants, MSTCN over Transformer) are chosen with this budget in mind, not
  a larger assumed GPU.
- Raw data (REASSEMBLE and any other datasets pulled in) is gitignored, matching
  `CLAUDE.md`'s existing instruction not to commit raw data.

## Estimates (rough, to confirm once real data is inspected)

Given REASSEMBLE's approximate scale (~148 demos) and the 8GB GPU:
- Tokenizer training: ~15-40 min.
- Telemetry-only classifier training: ~15-30 min.
- Vision feature extraction (frozen, one-time, cacheable): ~20-40 min.
- Vision-enriched classifier training: ~15-30 min.
- Total GPU compute for one full pass: roughly 1.5-3 hours, likely spread across multiple
  sessions in practice for debugging/hyperparameter iteration.

These were order-of-magnitude guesses based on typical scale for this model size at this
demo count, made before inspecting real data. Now confirmed: demos run ~100s+ each at
~1000Hz telemetry (see Confirmed facts below) — more windows per demo than assumed, which
likely keeps tokenizer training time in the same ballpark (window count drives batches/epoch,
not wall-clock per sample meaningfully at this model size) but should be re-measured once
the full 148-demo set is available rather than trusted as-is.

## Confirmed facts from real data inspection (supersedes earlier assumptions)

Verified by downloading and inspecting an actual REASSEMBLE demo file
(`2025-01-09-13-57-17.h5`) rather than relying on the ~20Hz precedent assumption:

- **Telemetry rate is ~970-1000Hz** (joint positions/velocities/efforts, gripper state,
  measured force/torque), **not ~20Hz**. Pose/velocity run ~570Hz. This is far denser than
  assumed — good for the tokenizer (more windows of training signal per demo) but the
  window-length hyperparameter should be chosen in raw-sample terms accordingly (e.g. a
  150ms window is ~150 samples at this rate, not ~3).
- **Video is standard 30fps** (hama1/hama2 static cameras, ~29fps hand camera). There is
  also a separate **event camera**: an async sparse event stream (~90k events/sec, not
  frame-based) plus a distinct 6Hz RGB snapshot stream from the same physical camera. Event
  data is out of scope for v1 (not part of the tokenizer/vision-enrichment design) but noted
  here in case it becomes relevant later.
- **REASSEMBLE ships two label granularities**, not one: **high-level** segments
  (task/object-specific, e.g. "Pick Ethernet", "Insert Ethernet", each with a success flag)
  and, nested inside each, **low-level segments** (e.g. Grasp, Lift, Approach, Align). The
  low-level labels are close to exactly the closed-vocabulary manipulation primitives
  `HANDOFF.md` describes (task/object-agnostic, expected to recur across demos) — **the
  temporal classifier should target low-level labels**, not high-level ones, since
  high-level labels are tied to specific objects/tasks and won't generalize the way the
  project's vocabulary is meant to. High-level segments remain useful as a coarser
  auxiliary signal (e.g. sanity-checking that low-level predictions nest correctly inside
  high-level ones) but are not the primary classification target.
- The 149-file TUData `data.zip` deposit is *already* scoped to the paper's official
  148-demo benchmark split (`train_split1.txt` + `test_split1.txt`) — the one extra file
  (`2025-01-10-16-17-40`) is the demo the dataset's own known-issues notes flag as having a
  missing hand camera, which is why it's excluded from the split. No need to source the
  larger 4,551-demo raw collection mentioned in the paper abstract.
- **Download bandwidth from the TUData host is slow (~1-1.5MB/s measured, both via the
  download script and raw `curl`), making the full 148-file / ~58.9GB set a ~10-11 hour
  download.** Not a code issue — appears to be server-side throttling. For this pass, only
  a 10-demo subset was downloaded (`data/reassemble/`, gitignored) to unblock writing and
  testing tokenizer/data-loading code without waiting on the full set. Downloading the
  remaining demos is deferred — needed before the real F1@50 milestone (item 5 in Data flow
  & milestones) but not before that.

## Open questions / assumptions carried into implementation

- Tokenizer hyperparameters (window length, codebook size) — starting values given above,
  expected to be tuned empirically rather than fixed by this spec.
- Whether any additional open dataset gets mixed into tokenizer pretraining — left open,
  decided opportunistically during implementation if it looks likely to help.
- TCN vs. Transformer for the classifier is a staged decision (TCN first), not a permanent
  one — revisit if TCN can't capture sufficient context once measured on REASSEMBLE.
- Full 148-demo download (needed for the real F1@50 milestone) has not run yet — only a
  10-demo subset is present locally as of this writing.
