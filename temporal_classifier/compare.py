"""Run telemetry-only vs. vision-enriched classifier training and report the F1@50 delta."""

import os

from temporal_classifier.train import run_training
from tokenizer.data import load_demo, resample_to_grid
from tokenizer.evaluate import check_tokenizer_gates, evaluate_tokenizer
from tokenizer.train import load_checkpoint, train_tokenizer
from tokenizer.windowing import split_available_demos_3way


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
         skip_gates: bool = False):
    from enrichment.vision_features import load_vision_encoder

    train_paths, val_paths, test_paths = split_available_demos_3way()
    train_paths, eval_paths, split_name = resolve_eval_split(train_paths, val_paths, test_paths, eval_on)
    print(f"{len(train_paths)} train demos, {len(eval_paths)} {split_name} demos available")
    if split_name == "test":
        print("!! scoring on the held-out test split -- only do this once, with hyperparameters frozen")

    if fresh and reset_tokenizer_checkpoint(tokenizer_ckpt):
        print(f"removed stale tokenizer checkpoint at {tokenizer_ckpt}; training from scratch")
    train_tokenizer(train_paths, tokenizer_ckpt, device="cuda")

    # The spec's checks 1-3 gate moving on to the classifier. Run them on held-out demos
    # before spending hours of GPU time on three classifier arms built over a bad tokenizer.
    model, mean, std, config = load_checkpoint(tokenizer_ckpt)
    report = evaluate_tokenizer(
        model, config["num_codes"], config["window"], _load_demos(eval_paths), mean, std, device="cuda"
    )
    print(format_tokenizer_report(report))
    failures = check_tokenizer_gates(report)
    if failures:
        for f in failures:
            print(f"  GATE FAILED: {f}")
        if not skip_gates:
            raise SystemExit(
                "tokenizer failed its quality gates; fix the tokenizer before training the "
                "classifier, or pass --skip-gates to proceed anyway"
            )
        print("  proceeding anyway (--skip-gates)")

    telemetry_only = run_training(tokenizer_ckpt, train_paths, eval_paths, device="cuda")
    vision_encoder = load_vision_encoder(device="cuda")

    # All arms share one tokenizer checkpoint and the control arm's vocabulary, so the only
    # thing that varies between them is the vision pathway.
    vision_kwargs = dict(
        vocab=telemetry_only["vocab"],
        device="cuda",
        use_vision=True,
        vision_encoder=vision_encoder,
        vision_cache_dir="vision_cache",
    )
    vision_enriched = run_training(tokenizer_ckpt, train_paths, eval_paths, **vision_kwargs)
    vision_pooled = run_training(
        tokenizer_ckpt, train_paths, eval_paths, pool_vision_over_window=True, **vision_kwargs
    )

    print(
        format_comparison(
            telemetry_only["val_f1"], vision_enriched["val_f1"], vision_pooled["val_f1"]
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-on", choices=["val", "test"], default="val")
    parser.add_argument("--keep-checkpoint", action="store_true", help="reuse an existing tokenizer checkpoint")
    parser.add_argument("--skip-gates", action="store_true", help="train the classifier even if the tokenizer fails its gates")
    args = parser.parse_args()
    main(eval_on=args.eval_on, fresh=not args.keep_checkpoint, skip_gates=args.skip_gates)
