# enrichment

Attaches a time-aligned frozen visual feature (DINOv2/CLIP) to each motion token via
concatenate+project or additive fusion. Cross-attention is a fallback, not built in v1.

See `docs/superpowers/specs/2026-09-08-phase2-core-motion-tokenizer-design.md`.
