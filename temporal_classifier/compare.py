"""Run telemetry-only vs. vision-enriched classifier training and report the F1@50 delta."""

from temporal_classifier.train import run_training
from tokenizer.train import train_tokenizer
from tokenizer.windowing import split_available_demos


def format_comparison(telemetry_only_f1: float, vision_enriched_f1: float) -> str:
    delta = vision_enriched_f1 - telemetry_only_f1
    sign = "+" if delta >= 0 else ""
    return (
        "F1@50 comparison\n"
        "-----------------\n"
        f"telemetry-only:   {telemetry_only_f1:.2f}\n"
        f"vision-enriched:  {vision_enriched_f1:.2f}\n"
        f"delta:            {sign}{delta:.2f}\n"
    )


def main():
    from enrichment.vision_features import load_vision_encoder

    train_paths, val_paths = split_available_demos()
    print(f"{len(train_paths)} train demos, {len(val_paths)} val demos available")

    tokenizer_ckpt = "tokenizer_checkpoint.pt"
    train_tokenizer(train_paths, tokenizer_ckpt, device="cuda")

    telemetry_only = run_training(tokenizer_ckpt, train_paths, val_paths, device="cuda")
    vision_encoder = load_vision_encoder(device="cuda")
    vision_enriched = run_training(
        tokenizer_ckpt,
        train_paths,
        val_paths,
        vocab=telemetry_only["vocab"],
        device="cuda",
        use_vision=True,
        vision_encoder=vision_encoder,
    )

    print(format_comparison(telemetry_only["val_f1"], vision_enriched["val_f1"]))


if __name__ == "__main__":
    main()
