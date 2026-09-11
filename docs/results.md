# Results index

Measured F1@50 numbers live in versioned folders under `results/`, one per run that produced
a usable result. Each holds the write-up, the figures, and the raw artifacts that produced
them.

| version | run | scored on | headline |
|---|---|---|---|
| [v1](../results/v1/README.md) | `20260911-114547` | validation (22 demos) | telemetry-only **0.8419**, vision 1-frame/token **0.7746** (**-0.067**); pooled arm pending |

See [are-our-results-good.md](are-our-results-good.md) for why vision appears to hurt, how
these numbers sit against M2R2 and Nomadic, and what would be needed to claim anything.

Aborted and failed attempts are not given a version. They stay in `runs/` with a
`status.json` recording what went wrong — see the "Superseded runs" section of each version's
write-up for the short history.

**Nothing here is comparable to M2R2 or Nomadic yet.** Every number so far is on the
validation split; `test_split1` is untouched and should be scored once, with hyperparameters
frozen (`--eval-on test`).
