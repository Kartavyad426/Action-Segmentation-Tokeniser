"""Run telemetry-only vs. vision-enriched classifier training and report the F1@50 delta."""

from temporal_classifier.train import run_training
from tokenizer.train import train_tokenizer
from tokenizer.windowing import split_available_demos


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


def main():
    from enrichment.vision_features import load_vision_encoder

    train_paths, val_paths = split_available_demos()
    print(f"{len(train_paths)} train demos, {len(val_paths)} val demos available")

    tokenizer_ckpt = "tokenizer_checkpoint.pt"
    train_tokenizer(train_paths, tokenizer_ckpt, device="cuda")

    telemetry_only = run_training(tokenizer_ckpt, train_paths, val_paths, device="cuda")
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
    vision_enriched = run_training(tokenizer_ckpt, train_paths, val_paths, **vision_kwargs)
    vision_pooled = run_training(
        tokenizer_ckpt, train_paths, val_paths, pool_vision_over_window=True, **vision_kwargs
    )

    print(
        format_comparison(
            telemetry_only["val_f1"], vision_enriched["val_f1"], vision_pooled["val_f1"]
        )
    )


if __name__ == "__main__":
    main()
