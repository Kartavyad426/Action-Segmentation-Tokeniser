"""Run telemetry-only vs. vision-enriched classifier training and report the F1@50 delta."""

import faulthandler
import json
import logging
import os
import signal
import shutil
import subprocess
import sys
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
    epochs=20, window=15, stride=5, latent_dim=32, num_codes=512, hidden=64, batch_size=64,
    lr=1e-3, seed=0,
)

log = logging.getLogger("compare")


@dataclass
class RunPaths:
    """Everything a single run produces, all under one directory that is never reused."""

    dir: str
    log: str
    config: str
    checkpoint: str
    results: str
    tokenizer_report: str
    tokenizer_history: str


def new_run(base: str = "runs", now: datetime | None = None) -> RunPaths:
    """Create a fresh timestamped run directory and point `latest` at it.

    Runs are never overwritten: the checkpoint, the config, the log and the results all live
    together, so a run stays interpretable long after it finished and a later run cannot
    destroy an earlier one's evidence.
    """
    now = now or datetime.now()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(base, stamp)
    suffix = 1
    while os.path.exists(path):  # same-second reruns must not collide
        path = os.path.join(base, f"{stamp}-{suffix}")
        suffix += 1
    os.makedirs(path)

    link = os.path.join(base, "latest")
    tmp_link = link + ".tmp"
    if os.path.islink(tmp_link) or os.path.exists(tmp_link):
        os.remove(tmp_link)
    os.symlink(os.path.abspath(path), tmp_link)
    os.replace(tmp_link, link)

    return RunPaths(
        dir=path,
        log=os.path.join(path, "run.log"),
        config=os.path.join(path, "config.json"),
        checkpoint=os.path.join(path, "tokenizer_checkpoint.pt"),
        results=os.path.join(path, "results.json"),
        tokenizer_report=os.path.join(path, "tokenizer_report.json"),
        tokenizer_history=os.path.join(path, "tokenizer_history.json"),
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_run_config(run: RunPaths, **info) -> dict:
    """Record everything needed to interpret or reproduce this run's numbers later."""
    import torch as _torch

    splits = info.pop("splits", {})
    config = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "torch_version": _torch.__version__,
        "python_version": sys.version.split()[0],
        "command": " ".join(sys.argv),
        "run_dir": run.dir,
        # demo membership, not just counts: a split that silently shifts must be detectable
        "splits": {**splits, "counts": {k: len(v) for k, v in splits.items()}},
        **info,
    }
    with open(run.config, "w") as f:
        json.dump(config, f, indent=2)
    return config


def adopt_previous_run(run: RunPaths, previous_dir: str, fingerprint: str) -> bool:
    """Copy a previous run's tokenizer and completed arms into this run, if compatible.

    Copied, never referenced in place: the new run must own a complete, self-contained
    record, and the previous run must remain exactly as it was.
    """
    old = RunPaths(
        dir=previous_dir,
        log=os.path.join(previous_dir, "run.log"),
        config=os.path.join(previous_dir, "config.json"),
        checkpoint=os.path.join(previous_dir, "tokenizer_checkpoint.pt"),
        results=os.path.join(previous_dir, "results.json"),
        tokenizer_report=os.path.join(previous_dir, "tokenizer_report.json"),
        tokenizer_history=os.path.join(previous_dir, "tokenizer_history.json"),
    )
    # Prefer the previous run's best held-out checkpoint: that is the one compare.py would
    # itself have selected, and a run killed mid-training leaves a last-epoch checkpoint
    # that is often worse than the best epoch it already saw.
    best = old.checkpoint[:-3] + ".best.pt"
    source = best if os.path.exists(best) else old.checkpoint
    if not os.path.exists(source):
        return False
    try:
        import torch as _torch

        found = _torch.load(source, map_location="cpu", weights_only=False)["config"].get("fingerprint")
    except Exception:
        return False
    if found != fingerprint:
        return False

    shutil.copy2(source, run.checkpoint)
    if os.path.exists(old.results):
        shutil.copy2(old.results, run.results)
    return True


def install_hang_diagnostics(run: RunPaths, watchdog_seconds: int = 900) -> str:
    """Make a hang self-documenting.

    Two runs stalled with the process pinned at 100% CPU and no output, and there was no way
    to see where: yama ptrace_scope=1 blocks attaching a profiler to a process that is not a
    descendant of the attaching shell. faulthandler works from inside the process instead.

    - SIGUSR1 dumps every thread's Python stack on demand (`kill -USR1 <pid>`).
    - A repeating watchdog dumps automatically if no epoch completes within the timeout, so
      an unattended overnight hang still leaves evidence.
    """
    stream = open(os.path.join(run.dir, "faulthandler.log"), "w", buffering=1)
    faulthandler.enable(file=stream, all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=stream, all_threads=True, chain=True)
    faulthandler.dump_traceback_later(watchdog_seconds, repeat=True, file=stream, exit=False)
    return stream.name


def setup_logging(run_dir: str) -> str:
    """Everything printed also lands in a file, so a run that dies leaves a record."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "run.log")
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    # Configure the ROOT logger. tokenizer.train and temporal_classifier.train report their
    # own per-epoch and per-demo progress through module loggers, and all of it has to reach
    # the same file. Mixing print() with logging previously sent that progress to whatever
    # stdout happened to be -- on a nohup run, /dev/null.
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for handler in (logging.StreamHandler(), logging.FileHandler(path)):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    return path


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


CLASSIFIER_HP = dict(epochs=30, channels=64, num_layers=9, num_stages=3, lr=1e-3,
                     smoothing_weight=0.15, tau=4.0, fusion_out_dim=128, eval_every=5, seed=0)


def main(eval_on: str = "val", runs_base: str = "runs", resume_from: str | None = None,
         skip_gates: bool = False, tokenizer_seed: int | None = None,
         classifier_seed: int | None = None):
    from enrichment.vision_features import load_vision_encoder

    tokenizer_hp = dict(TOKENIZER_HP)
    classifier_hp = dict(CLASSIFIER_HP)
    if tokenizer_seed is not None:
        tokenizer_hp["seed"] = tokenizer_seed
    if classifier_seed is not None:
        classifier_hp["seed"] = classifier_seed

    run = new_run(runs_base)
    setup_logging(run.dir)
    dump_path = install_hang_diagnostics(run)
    started = time.time()
    log.info(f"run directory {run.dir}")
    log.info(f"hang diagnostics: kill -USR1 {os.getpid()} dumps stacks to {dump_path}")

    train_paths, val_paths, test_paths = split_available_demos_3way()
    # Held-out set for the tokenizer's own per-epoch loss. Only available when scoring on
    # validation: the final test run folds val into training, and test must not be touched.
    tokenizer_val = val_paths if eval_on == "val" else None
    splits = {"train": train_paths, "val": val_paths, "test": test_paths}
    train_paths, eval_paths, split_name = resolve_eval_split(train_paths, val_paths, test_paths, eval_on)
    log.info(f"{len(train_paths)} train demos, {len(eval_paths)} {split_name} demos")
    if split_name == "test":
        log.info("!! scoring on the held-out test split -- only do this once, hyperparameters frozen")

    # The tokenizer seed is part of the fingerprint (a different seed is a different
    # tokenizer); the classifier seed is not, so classifier repeats can reuse one tokenizer.
    fingerprint = tokenizer_fingerprint(train_paths, **tokenizer_hp)
    config = write_run_config(
        run, eval_on=eval_on, split_scored=split_name, fingerprint=fingerprint,
        tokenizer_hp=tokenizer_hp, classifier_hp=classifier_hp, splits=splits,
        vision_cache_dir="vision_cache", resume_from=resume_from,
    )
    log.info(f"config written to {run.config} (git {config['git_commit'][:8]}, torch {config['torch_version']})")
    log.info(f"tokenizer fingerprint {fingerprint}")

    adopted = adopt_previous_run(run, resume_from, fingerprint) if resume_from else False
    if resume_from:
        log.info(f"resume from {resume_from}: {'adopted tokenizer + results' if adopted else 'REFUSED (missing or fingerprint mismatch) -- training fresh'}")

    if not adopted:
        log.info(f"training tokenizer on {len(train_paths)} demos, {TOKENIZER_HP['epochs']} epochs")
        def save_history(epoch, history):
            """Written every epoch so a run in progress can be plotted and inspected."""
            with open(run.tokenizer_history, "w") as f:
                json.dump({"fingerprint": fingerprint, "history": history}, f, indent=2)

        result = train_tokenizer(
            train_paths, run.checkpoint, device="cuda", val_paths=tokenizer_val,
            on_epoch=save_history, **tokenizer_hp
        )
        with open(run.tokenizer_history, "w") as f:
            json.dump({
                "fingerprint": fingerprint, "history": result["history"],
                "best_epoch": result["best_epoch"], "best_val_loss": result["best_val_loss"],
                "best_utilization": result["best_utilization"],
            }, f, indent=2)
        if result["best_checkpoint"]:
            log.info(
                f"best codebook utilization {result['best_utilization']:.3f} at epoch "
                f"{result['best_epoch']}/{TOKENIZER_HP['epochs']} "
                f"(held-out loss {result['best_val_loss']:.5f}); using that checkpoint "
                f"rather than the last -- loss alone selects for codebook collapse"
            )
            shutil.copy2(result["best_checkpoint"], run.checkpoint)
        log.info(f"tokenizer ready at {run.checkpoint}")

    # The spec's checks 1-3 gate moving on to the classifier. Run them on held-out demos
    # before spending hours of GPU time on three classifier arms built over a bad tokenizer.
    model, mean, std, tok_config = load_checkpoint(run.checkpoint)
    report = evaluate_tokenizer(
        model, tok_config["num_codes"], tok_config["window"], _load_demos(eval_paths), mean, std, device="cuda"
    )
    for line in format_tokenizer_report(report).rstrip().splitlines():
        log.info(line)
    with open(run.tokenizer_report, "w") as f:
        json.dump({"fingerprint": fingerprint, **report}, f, indent=2)

    failures = check_tokenizer_gates(report)
    if failures:
        for failure in failures:
            log.info(f"  GATE FAILED: {failure}")
        if not skip_gates:
            raise SystemExit(
                "tokenizer failed its quality gates; fix the tokenizer before training the "
                "classifier, or pass --skip-gates to proceed anyway"
            )
        log.info("  proceeding anyway (--skip-gates)")

    results = ResultLog(run.results, fingerprint)
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
        log.info(f"arm {name}: starting")
        t0 = time.time()
        out = run_training(
            run.checkpoint, train_paths, eval_paths, vocab=vocab, device="cuda",
            vision_encoder=vision_encoder, vision_cache_dir="vision_cache", **extra, **classifier_hp,
        )
        vocab = vocab or out["vocab"]
        results.record(
            name,
            {"val_f1": out["val_f1"], "seconds": round(time.time() - t0, 1),
             "history": out["history"], "vocab": out["vocab"]},
            fingerprint,
        )
        log.info(f"arm {name}: F1@50 {out['val_f1']:.4f}  ({time.time() - t0:.0f}s)  -> {run.results}")

    for line in format_comparison(
        results.get("telemetry_only")["val_f1"],
        results.get("vision_nearest")["val_f1"],
        results.get("vision_pooled")["val_f1"],
    ).rstrip().splitlines():
        log.info(line)
    log.info(f"run finished in {(time.time() - started) / 60:.1f} min -- all artifacts in {run.dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-on", choices=["val", "test"], default="val")
    parser.add_argument("--skip-gates", action="store_true", help="train the classifier even if the tokenizer fails its gates")
    parser.add_argument("--runs-base", default="runs", help="parent directory for per-run folders")
    parser.add_argument("--tokenizer-seed", type=int, default=None,
                        help="seed for tokenizer training; part of the fingerprint")
    parser.add_argument("--classifier-seed", type=int, default=None,
                        help="seed for classifier training; NOT part of the fingerprint, so "
                             "repeats can reuse one tokenizer via --resume-from")
    parser.add_argument("--resume-from", default=None,
                        help="a previous run directory whose tokenizer and completed arms to reuse "
                             "(only if its fingerprint matches this run's configuration)")
    args = parser.parse_args()
    main(eval_on=args.eval_on, runs_base=args.runs_base, resume_from=args.resume_from,
         skip_gates=args.skip_gates, tokenizer_seed=args.tokenizer_seed,
         classifier_seed=args.classifier_seed)
