# temporal_classifier

MSTCN-style TCN reading the full enriched token sequence (before/after context) to predict
per-token action labels. Transformer is a fallback upgrade if TCN can't capture enough
context, not built in v1.

See `docs/superpowers/specs/2026-09-08-phase2-core-motion-tokenizer-design.md`.
