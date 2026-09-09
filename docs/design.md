# Phase 2 Core: Design & Implementation

Living document — describes what's actually built and how it works. Update this whenever
the implementation changes. For the *differences from published reference results*, see
`docs/differences-from-published-results.md` instead — this doc is "what we built and why,"
that one is "how it might diverge from what Nomadic/M2R2 report."

Source of truth for original intent: `docs/superpowers/specs/2026-09-08-phase2-core-motion-tokenizer-design.md`
(the approved spec) and `docs/superpowers/plans/2026-09-08-phase2-core-motion-tokenizer.md`
(the task-by-task implementation plan). This doc reflects current reality, which may have
moved past those as bugs were found and fixed — check git log / the plan's ledger
(`.superpowers/sdd/2026-09-08-phase2-core-motion-tokenizer/progress.md`) for the detailed
history of what changed and why.

## What this is

A three-stage pipeline validated on the REASSEMBLE open dataset (148-demo official
benchmark split), built to answer one question before touching real robot data: does vision
enrichment actually help temporal action segmentation in our implementation, the way
Nomadic AI's blog claims (79.5% -> 93.1% F1@50) and unlike M2R2's own ablation (74.5% ->
74.6%)?

```
telemetry (.h5)  --resample to 100Hz-->  windows  --VQ-VAE-->  motion tokens
                                                                      |
video (.h5, same file) --nearest frame--> DINOv2 --> vision feature -+--concat+project--> fused token
                                                                      |
                                                              MS-TCN classifier
                                                                      |
                                                        per-token label (Grasp/Lift/Approach/Align/...)
```

## Components

### `tokenizer/` — motion tokenizer

- `data.py` — loads one REASSEMBLE `.h5` demo: 5 telemetry channels
  (`joint_positions`, `joint_velocities`, `gripper_positions`, `measured_force`,
  `measured_torque`; 22 dims total) plus the low-level action segments
  (`(start, end, text)` tuples, e.g. `(1736427449.51, 1736427460.14, "Grasp")`). Resamples
  every channel (native rate confirmed ~970-1000Hz) onto one shared **100Hz** grid via
  linear interpolation — this is the point where per-channel native-rate telemetry becomes
  one aligned multi-channel array.
- `windowing.py` — slices the resampled array into fixed windows. Two modes: overlapping
  (window=15 samples/150ms, stride=5/50ms) for VQ-VAE training diversity, and
  non-overlapping (stride=window) for producing one clean token per demo timestep when
  building a classifier input sequence.
- `model.py` — `MotionTokenizer`: conv1d encoder -> vector-quantization bottleneck (512
  codes, 32-dim latent) -> conv1d decoder. Trains on reconstruction + VQ losses only, no
  labels.
- `train.py` — training loop; saves a checkpoint with model weights, normalization
  stats, and config.
- `evaluate.py` — three label-free health checks: held-out reconstruction error, codebook
  utilization (usage entropy, catches codebook collapse), boundary-alignment score (do token
  changes cluster near ground-truth segment boundaries).

### `enrichment/` — vision features

- `vision_features.py` — frozen DINOv2 (`dinov2_vits14`, 384-dim CLS token) run on the
  video frame nearest each token's center timestamp. Preprocessing: resize-shortest-side +
  center-crop to 224x224, ImageNet mean/std normalization (fixed after final review found
  the original implementation squashed aspect ratio and skipped normalization entirely).
  - Extraction streams: frames are decoded one at a time, preprocessed into a batch buffer,
    encoded in batches of 32, and discarded. Peak RAM is O(batch size), not O(demo length)
    — measured flat at ~85 MB from 300 to 1200 frames, against ~568 MB at 600 frames for the
    original decode-everything-first approach.
  - `cache_dir` persists features to disk, keyed on demo + camera + a hash of the exact
    frame indices requested, so a window-config change misses the cache rather than serving
    features aligned to the old token grid. Wired through `run_training(vision_cache_dir=)`
    and used by `compare.py`.
- `fuse.py` — `ConcatProjectFusion`: LayerNorms the 32-dim motion embedding and the 384-dim
  vision feature **separately**, concatenates them, and projects to `out_dim` (default 128)
  via one learned linear layer. This projection trains jointly with the classifier — it's
  the only supervised part of the vision pathway.
  - The per-branch normalization is load-bearing, not hygiene: `nn.Linear` draws all 416
    input weights from one distribution, so each branch's contribution to the output scales
    with its vector norm. Unnormalized, vision entered ~11-26x louder than motion (the exact
    figure drifted with the tokenizer checkpoint) and swamped the motion signal at init.
    LayerNorm pins each branch to norm √dim, leaving only the √(384/32) = 3.46x that is pure
    dimensionality. Normalizing the concatenated 416-dim vector instead would apply one
    shared scalar to both halves and barely change the ratio — see the differences doc.
  - `out_dim` is independent of the tokenizer's `latent_dim` (it was previously pinned to
    it, coupling two unrelated hyperparameters and forcing 416 dims through a 32-dim
    bottleneck). `MSTCN`'s `in_channels` follows the fusion width on the vision arm and
    stays at `latent_dim` on the telemetry-only control arm.

### `temporal_classifier/` — the actual classifier

- `labels.py` — builds the closed vocabulary from REASSEMBLE's **low-level** segment
  labels specifically (Grasp, Lift, Approach, Align, ...) — not the high-level task labels
  ("Pick Ethernet"), which are object/task-specific and wouldn't generalize the way this
  project's vocabulary is meant to. `background` is always class 0.
- `model.py` — `MSTCN`: multi-stage TCN (3 stages, 9 dilated-conv layers/stage, 64
  channels), chosen over a Transformer per the spec's staged-complexity decision. Each stage
  refines the previous stage's softmax output.
- `metrics.py` — segmental F1@k (`f1_at_k`), the standard TAS evaluation metric
  (Lea et al. protocol: match predicted/ground-truth segments by IoU >= threshold, same
  label, greedy one-to-one matching). Also `f1_at_k_corpus` — aggregates TP/FP/FN across
  multiple demos before computing one F1, matching how the Nomadic/M2R2 reference numbers
  are actually computed (see differences doc — the original implementation macro-averaged
  per-demo instead, which is not the same number).
- `train.py` — trains `MSTCN` on frozen tokenizer token embeddings, optionally fused with
  vision features (`use_vision=True`). Same function handles both the telemetry-only
  baseline and the vision-enriched run.
- `compare.py` — runs both modes back to back and prints the F1@50 delta. This is the
  actual go/no-go deliverable.

See `docs/hyperparameters.md` for every hyperparameter, where it is set, and which ones
are worth changing.

## Key implementation decisions and why

- **100Hz resampling grid, not the originally-assumed ~20Hz.** Real REASSEMBLE telemetry
  runs ~970-1000Hz; confirmed by direct inspection, not left as an assumption.
- **Low-level labels as the classifier target**, not high-level. Confirmed by inspecting
  actual segment structure: low-level segments (Grasp/Lift/Approach/Align) are the
  primitives; high-level segments (Pick Ethernet, Insert Ethernet) are task/object-specific
  composites.
- **MS-TCN over Transformer**, staged decision — cheaper to train on the available 8GB GPU,
  proven at REASSEMBLE's data scale (M2R2 also uses MSTCN-family heads).
- **Concat+project vision fusion**, not cross-attention — matches the spec's staged
  approach (cross-attention is an explicit fallback, not built unless concat+project proves
  insufficient once measured).
- **`torch.set_num_threads(1)` on CPU device.** Found during Task 3 implementation:
  PyTorch's default CPU intra-op thread pool causes ~90-140x per-step overhead on this
  workload's tiny batch/model sizes (measured directly, not assumed). Applied consistently
  in both `tokenizer/train.py` and `temporal_classifier/train.py`, guarded on
  `device == "cpu"` so GPU runs are unaffected.

## Current status

All 10 implementation-plan tasks complete, full test suite passing. A final whole-branch
review found 4 blocking issues (device propagation bug in the vision path, missing test
coverage for vision-fusion training, incorrect DINOv2 preprocessing, F1 aggregation
convention mismatch vs. published numbers) — see the ledger
(`.superpowers/sdd/2026-09-08-phase2-core-motion-tokenizer/progress.md`) and the
differences doc for full detail and status.

Only 10 of the eventual 148 REASSEMBLE benchmark demos are downloaded as of this writing;
the rest are downloading in the background. No real benchmark F1@50 number exists yet —
everything so far is pipeline-correctness testing on the small subset. Per project memory
(`robotics-fresh-training-no-smoke-test-carryover`), the real training run on the full
148-demo set must not warm-start from any checkpoint produced during this testing phase.
