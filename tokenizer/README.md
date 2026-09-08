# tokenizer

VQ-VAE-style motion tokenizer. Encodes short windows of raw telemetry into discrete
motion-token IDs via a learned codebook. Trains on reconstruction only — no labels.

See `docs/superpowers/specs/2026-09-08-phase2-core-motion-tokenizer-design.md`.
