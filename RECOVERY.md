# If the v1 run dies before arm 3 finishes

Run `20260911-114547` produced `results/v1`. If its process is killed, suspended past
recovery, or the machine goes down mid-arm-3, this is how to continue **without losing arms 1
and 2 or changing their numbers**.

```bash
python -m temporal_classifier.compare \
    --eval-on val \
    --resume-from runs/20260911-114547 \
    --force-resume
```

`--force-resume` is required, and only for this run. The tokenizer seed was added to the
fingerprint *while this run was in flight*, so its checkpoint is stamped `c0c24bcfd26cbbe6`
while a new run computes `63662da6c5f7c949` from the same demos and hyperparameters. Without
the flag the guard refuses (correctly — it cannot know the difference is cosmetic), retrains
a fresh tokenizer, and discards `results.json` as belonging to a different tokenizer. You
would then re-run all three arms and get different numbers than v1.

## What survives a crash regardless

| | state |
|---|---|
| arms 1 and 2 F1@50 + full per-epoch history | `runs/20260911-114547/results.json` |
| the tokenizer itself | `runs/20260911-114547/tokenizer_checkpoint{,.best}.pt` |
| every DINOv2 feature extracted so far | `vision_cache/` — survives across runs, keyed on demo + camera + per-token frame grouping, not on the tokenizer |
| run config, logs, gate report | `runs/20260911-114547/` |
| the published result | `results/v1/` — committed to git |

Arm 3 restarts from the beginning of *its own* extraction loop, but every demo it already
processed is a cache hit, so it re-does seconds of work per completed demo rather than
minutes.

## What is NOT preserved

The partially-trained arm-3 classifier. Classifier training is ~2 minutes once features are
cached, so this costs little.

## Future runs do not need any of this

`--force-resume` exists only because a fingerprint input changed mid-run. Runs started after
commit `fec9981` have the seed in their fingerprint from the beginning, so a plain
`--resume-from` will match and work.
