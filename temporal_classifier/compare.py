"""Run telemetry-only vs. vision-enriched classifier training and report the F1@50 delta."""

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime

import torch

from temporal_classifier.train import run_training
from tokenizer.data import load_demo, resample_to_grid
from tokenizer.evaluate import check_tokenizer_gates, evaluate_tokenizer
from tokenizer.train import load_checkpoint, tokenizer_fingerprint, train_tokenizer
from tokenizer.windowing import split_available_demos_3way


# One source of truth: these are passed to train_tokenizer *and* hashed into the
# fingerprint, so the recorded provenance cannot drift from what was actually trained.
TOKENIZER_HP = dict(
    epochs=20, window=15, stride=5, latent_dim=32, num_codes=512, hidden=64, batch_size=64, lr=1e-3,
)

log = logging.getLogger("compare")


def setup_logging(run_dir: str) -> str:
    """Everything printed also lands in a file, so a run that dies leaves a record."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "run.log")
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for handler in (logging.StreamHandler(), logging.FileHandler(path)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return path


@dataclass
class TokenizerDecision:
    retrain: bool
    reason: str


def resolve_tokenizer(path: str, expected_fingerprint: str, keep: bool = False) -> TokenizerDecision:
    """Decide whether an existing tokenizer checkpoint may be reused, and delete it if not.

    Reusing a checkpoint is only ever safe when it was trained on the same demos with the
    same hyperparameters. Anything else -- a changed codebook size, a changed demo split,
    a leftover from a smoke test -- would silently apply the wrong codebook and
    normalization statistics to the whole run, with nothing downstream able to detect it.
    So the stale file is removed rather than merely ignored: it cannot then be picked up by
    a later run either.
    """
    if not os.path.exists(path):
        return TokenizerDecision(True, "no checkpoint on disk")
    if not keep:
        os.remove(path)
        return TokenizerDecision(True, "training from scratch (pass --keep-checkpoint to reuse)")

    found = None
    try:
        found = torch.load(path, map_location="cpu", weights_only=False)["config"].get("fingerprint")
    except Exception as exc:  # unreadable or pre-fingerprint checkpoint
        os.remove(path)
        return TokenizerDecision(True, f"checkpoint unreadable ({type(exc).__name__}); removed")

    if found != expected_fingerprint:
        os.remove(path)
        return TokenizerDecision(
            True,
            f"fingerprint mismatch (checkpoint {found}, this run {expected_fingerprint}); removed",
        )
    return TokenizerDecision(False, f"reusing checkpoint with matching fingerprint {found}")


class ResultLog:
    """Per-arm results written to disk the moment each arm finishes.

    Arms take tens of minutes each and are only comparable when they share one tokenizer, so
    results are keyed by the tokenizer fingerprint: a retrained tokenizer invalidates every
    arm recorded before it.
    """

    def __init__(self, path: str, fingerprint: str | None = None):
        self.path = path
        self.fingerprint = fingerprint
        self.arms = {}
        if os.path.exists(path):
            try:
                saved = json.loads(open(path).read())
            except (json.JSONDecodeError, OSError):
                saved = {}
            if fingerprint is None or saved.get("fingerprint") == fingerprint:
                self.arms = saved.get("arms", {})

    def get(self, arm: str):
        return self.arms.get(arm)

    def record(self, arm: str, result: dict, fingerprint: str | None = None):
        self.fingerprint = fingerprint or self.fingerprint
        self.arms[arm] = result
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as f:
            json.dump({"fingerprint": self.fingerprint, "arms": self.arms}, f, indent=2)
        os.replace(tmp, self.path)  # atomic: a crash mid-write cannot corrupt the log


def reset_tokenizer_checkpoint(path: str) -> bool:
    """Delete a stale tokenizer checkpoint so a run always trains from scratch.

    `load_checkpoint` will happily reload whatever is at this path, applying an old
    codebook and old normalization stats to a new corpus. A checkpoint left behind by a
    smoke test on a handful of demos must not silently become the tokenizer for the full run.
    """
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def resolve_eval_split(train_paths, val_paths, test_paths, eval_on: str = "val"):
    """Which demos to train on and which to score against.

    Default is the validation split: `test_split1` is REASSEMBLE's published held-out set,
    and any number quoted against M2R2 or Nomadic has to come from a test set that nothing
    was tuned on. Selecting `test` folds the validation demos back into training, so the
    final run uses all 111 official train demos.
    """
    if eval_on == "test":
        return list(train_paths) + list(val_paths), list(test_paths), "test"
    if eval_on != "val":
        raise ValueError(f"eval_on must be 'val' or 'test', got {eval_on!r}")
    return list(train_paths), list(val_paths), "validation"


def format_comparison(
    telemetry_only_f1: float,
    vision_enriched_f1: float,
    vision_pooled_f1: float | None = None,
) -> str:
    def row(label: str, score: float) -> str:
        delta = score - telemetry_only_f1
        return f"{label:<28} {score:.2f}   {delta:+.2f}\n"

    report = (
        "F1@50 comparison\n"
        "----------------------------------------------\n"
        f"{'arm':<28} {'F1@50':>5}   {'delta':>5}\n"
        f"{'telemetry-only (control)':<28} {telemetry_only_f1:.2f}   {'--':>5}\n"
        + row("vision, 1 frame/token", vision_enriched_f1)
    )
    if vision_pooled_f1 is not None:
        report += row("vision, pooled over window", vision_pooled_f1)
    return report


def _load_demos(paths):
    demos = []
    for path in paths:
        channel_arrays, channel_timestamps, segments = load_demo(path)
        telemetry, grid = resample_to_grid(channel_arrays, channel_timestamps)
        demos.append((telemetry, grid, segments))
    return demos


def format_tokenizer_report(report: dict) -> str:
    b = report["boundary"]
    return (
        f"tokenizer health on {report['n_demos']} held-out demos\n"
        f"  reconstruction error   {report['reconstruction_error']:.4f}\n"
        f"  codebook utilization   {report['codebook_utilization']:.3f}"
        f"  (~{report['effective_codes']:.0f} effective codes)\n"
        f"  boundary f1            {b['f1']:.3f}"
        f"  (recall {b['boundary_recall']:.3f}, precision {b['change_precision']:.3f})\n"
        f"  boundary lift          {b['lift_over_chance']:+.3f}"
        f"  (chance {b['chance_recall']:.3f})\n"
        f"  segment purity         {b['segment_purity']:.3f}\n"
    )


def main(eval_on: str = "val", fresh: bool = True, tokenizer_ckpt: str = "tokenizer_checkpoint.pt",
         skip_gates: bool = False, run_dir: str = "runs/latest"):
    from enrichment.vision_features import load_vision_encoder

    log_path = setup_logging(run_dir)
    started = time.time()
    log.info(f"run started {datetime.now():%Y-%m-%d %H:%M:%S}  logging to {log_path}")

    train_paths, val_paths, test_paths = split_available_demos_3way()
    # Held-out set for the tokenizer's own per-epoch loss. Only available when scoring on
    # validation: the final test run folds val into training, and test must not be touched.
    tokenizer_val = val_paths if eval_on == "val" else None
    train_paths, eval_paths, split_name = resolve_eval_split(train_paths, val_paths, test_paths, eval_on)
    log.info(f"{len(train_paths)} train demos, {len(eval_paths)} {split_name} demos")
    if split_name == "test":
        log.info("!! scoring on the held-out test split -- only do this once, hyperparameters frozen")

    fingerprint = tokenizer_fingerprint(train_paths, **TOKENIZER_HP)
    decision = resolve_tokenizer(tokenizer_ckpt, fingerprint, keep=not fresh)
    log.info(f"tokenizer fingerprint {fingerprint}: {decision.reason}")

    if decision.retrain:
        log.info(f"training tokenizer on {len(train_paths)} demos")
        result = train_tokenizer(
            train_paths, tokenizer_ckpt, device="cuda", val_paths=tokenizer_val, **TOKENIZER_HP
        )
        for h in result["history"]:
            val = f"  val {h['val_loss']:.5f}" if h["val_loss"] is not None else ""
            log.info(f"  tokenizer epoch {h['epoch']:>3}/{TOKENIZER_HP['epochs']}  train {h['train_loss']:.5f}{val}")

    # The spec's checks 1-3 gate moving on to the classifier. Run them on held-out demos
    # before spending hours of GPU time on three classifier arms built over a bad tokenizer.
    model, mean, std, config = load_checkpoint(tokenizer_ckpt)
    report = evaluate_tokenizer(
        model, config["num_codes"], config["window"], _load_demos(eval_paths), mean, std, device="cuda"
    )
    for line in format_tokenizer_report(report).rstrip().splitlines():
        log.info(line)
    with open(os.path.join(run_dir, "tokenizer_report.json"), "w") as f:
        json.dump({"fingerprint": fingerprint, **report}, f, indent=2)

    failures = check_tokenizer_gates(report)
    if failures:
        for f in failures:
            log.info(f"  GATE FAILED: {f}")
        if not skip_gates:
            raise SystemExit(
                "tokenizer failed its quality gates; fix the tokenizer before training the "
                "classifier, or pass --skip-gates to proceed anyway"
            )
        log.info("  proceeding anyway (--skip-gates)")

    results = ResultLog(os.path.join(run_dir, "results.json"), fingerprint)
    vision_encoder = None

    # All arms share one tokenizer checkpoint and the control arm's vocabulary, so the only
    # thing that varies between them is the vision pathway.
    arms = [
        ("telemetry_only", {}),
        ("vision_nearest", {"use_vision": True}),
        ("vision_pooled", {"use_vision": True, "pool_vision_over_window": True}),
    ]
    vocab = None
    for name, extra in arms:
        done = results.get(name)
        if done is not None:
            log.info(f"arm {name}: already recorded, F1@50 {done['val_f1']:.4f} -- skipping")
            vocab = vocab or done.get("vocab")
            continue
        if extra.get("use_vision") and vision_encoder is None:
            vision_encoder = load_vision_encoder(device="cuda")
        log.info(f"arm {name}: training")
        t0 = time.time()
        out = run_training(
            tokenizer_ckpt, train_paths, eval_paths, vocab=vocab, device="cuda",
            vision_encoder=vision_encoder, vision_cache_dir="vision_cache", **extra,
        )
        vocab = vocab or out["vocab"]
        results.record(
            name,
            {"val_f1": out["val_f1"], "seconds": round(time.time() - t0, 1),
             "history": out["history"], "vocab": out["vocab"]},
            fingerprint,
        )
        log.info(f"arm {name}: F1@50 {out['val_f1']:.4f}  ({time.time() - t0:.0f}s)  -> results.json")

    report_text = format_comparison(
        results.get("telemetry_only")["val_f1"],
        results.get("vision_nearest")["val_f1"],
        results.get("vision_pooled")["val_f1"],
    )
    for line in report_text.rstrip().splitlines():
        log.info(line)
    log.info(f"run finished in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-on", choices=["val", "test"], default="val")
    parser.add_argument("--keep-checkpoint", action="store_true", help="reuse an existing tokenizer checkpoint")
    parser.add_argument("--skip-gates", action="store_true", help="train the classifier even if the tokenizer fails its gates")
    parser.add_argument("--run-dir", default="runs/latest", help="where run.log and results.json are written")
    args = parser.parse_args()
    main(eval_on=args.eval_on, fresh=not args.keep_checkpoint, skip_gates=args.skip_gates, run_dir=args.run_dir)
