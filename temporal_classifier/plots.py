"""Figures for a run directory: tokenizer convergence, classifier training, F1@50 delta.

Reads only the JSON a run already writes, so it can be pointed at a finished run, a run
still in progress, or a run that died early -- whatever artifacts exist get plotted.

    python -m temporal_classifier.plots runs/latest
"""

import json
import os
import re

import matplotlib

matplotlib.use("Agg")  # no display on a headless/background run
import matplotlib.pyplot as plt  # noqa: E402

ARM_LABELS = {
    "telemetry_only": "telemetry only",
    "vision_nearest": "vision, 1 frame/token",
    "vision_pooled": "vision, pooled over window",
}
ARM_COLORS = {"telemetry_only": "#4c72b0", "vision_nearest": "#dd8452", "vision_pooled": "#55a868"}


def tokenizer_series(history):
    """(epochs, train_losses, val_losses). val is empty when the run had no held-out demos."""
    epochs = [h["epoch"] for h in history]
    train = [h["train_loss"] for h in history]
    val = [h["val_loss"] for h in history if h.get("val_loss") is not None]
    return epochs, train, val


def classifier_series(history):
    """(epochs, losses, f1_epochs, f1s).

    Validation F1 is measured every `eval_every` epochs, so it carries its own epoch axis --
    zipping it against all epochs would silently misalign the points.
    """
    epochs = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]
    scored = [h for h in history if h.get("val_f1") is not None]
    return epochs, losses, [h["epoch"] for h in scored], [h["val_f1"] for h in scored]


_EPOCH_LINE = re.compile(
    r"tokenizer epoch\s+(\d+)/\d+\s+train\s+([\d.]+)(?:\s+val\s+([\d.]+))?"
)


def parse_tokenizer_log(log_path: str) -> list[dict]:
    """Recover the tokenizer curve from run.log.

    The log is the one artifact every run has, including runs that predate incremental
    history writing and runs killed between epochs.
    """
    history = []
    with open(log_path) as f:
        for line in f:
            m = _EPOCH_LINE.search(line)
            if m:
                epoch, train, val = m.groups()
                history.append({
                    "epoch": int(epoch),
                    "train_loss": float(train),
                    "val_loss": float(val) if val else None,
                })
    return history


def _load(run_dir, name):
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:  # a run still mid-write
        return None


def plot_tokenizer_training(data, out_path):
    epochs, train, val = tokenizer_series(data["history"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train, "-o", ms=3, label="train", color="#4c72b0")
    if val:
        ax.plot(epochs[: len(val)], val, "-o", ms=3, label="held-out validation", color="#c44e52")

    best = data.get("best_epoch")
    if best:
        ax.axvline(best, color="#555", ls="--", lw=1)
        ax.annotate(
            f"best held-out\nepoch {best}",
            xy=(best, max(train)), xytext=(4, -4), textcoords="offset points",
            fontsize=8, color="#555", va="top",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("VQ-VAE loss (recon + codebook + commitment)")
    ax.set_title("Tokenizer training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_classifier_training(results, out_path):
    arms = results["arms"]
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, arm in arms.items():
        history = arm.get("history") or []
        if not history:
            continue
        epochs, losses, f1_epochs, f1s = classifier_series(history)
        color = ARM_COLORS.get(name)
        label = ARM_LABELS.get(name, name)
        ax_loss.plot(epochs, losses, "-", label=label, color=color)
        ax_f1.plot(f1_epochs, f1s, "-o", ms=4, label=label, color=color)

    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("CE + 0.15 x T-MSE")
    ax_loss.set_title("Classifier training loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(fontsize=8)

    ax_f1.set_xlabel("epoch")
    ax_f1.set_ylabel("F1@50")
    ax_f1.set_title("Validation F1@50 during training")
    ax_f1.grid(alpha=0.3)
    ax_f1.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_f1_comparison(results, out_path, reference=None):
    arms = results["arms"]
    order = [k for k in ARM_LABELS if k in arms] + [k for k in arms if k not in ARM_LABELS]
    labels = [ARM_LABELS.get(k, k) for k in order]
    scores = [arms[k]["val_f1"] for k in order]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, scores, color=[ARM_COLORS.get(k, "#888") for k in order], width=0.55)
    baseline = arms.get("telemetry_only", {}).get("val_f1")
    for bar, name in zip(bars, order):
        score = arms[name]["val_f1"]
        delta = "" if baseline is None or name == "telemetry_only" else f"\n{score - baseline:+.3f}"
        ax.annotate(f"{score:.3f}{delta}", xy=(bar.get_x() + bar.get_width() / 2, score),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    if baseline is not None:
        ax.axhline(baseline, color="#4c72b0", ls="--", lw=1, alpha=0.6)

    # the whole point of the ablation is the delta against the control arm, so show it
    for label, value, color in (reference or []):
        ax.axhline(value, ls=":", lw=1, color=color)
        ax.annotate(label, xy=(len(labels) - 0.45, value), fontsize=7, color=color, va="bottom")

    ax.set_ylabel("F1@50")
    ax.set_title("Does vision help? (higher is better)")
    ax.set_ylim(0, max(scores + [v for _, v, _ in (reference or [])] + [0.1]) * 1.25)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_run(run_dir: str, out_dir: str | None = None) -> list[str]:
    """Write every figure the run has data for. Missing artifacts are skipped, not errors."""
    out_dir = out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)
    written = []

    tokenizer = _load(run_dir, "tokenizer_history.json")
    if not (tokenizer and tokenizer.get("history")):
        log_path = os.path.join(run_dir, "run.log")
        if os.path.exists(log_path):
            recovered = parse_tokenizer_log(log_path)
            if recovered:
                tokenizer = {"history": recovered, "best_epoch": None}
    if tokenizer and tokenizer.get("history"):
        written.append(plot_tokenizer_training(tokenizer, os.path.join(out_dir, "tokenizer_training.png")))

    results = _load(run_dir, "results.json")
    if results and results.get("arms"):
        if any(a.get("history") for a in results["arms"].values()):
            written.append(
                plot_classifier_training(results, os.path.join(out_dir, "classifier_training.png"))
            )
        written.append(plot_f1_comparison(results, os.path.join(out_dir, "f1_comparison.png")))

    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default="runs/latest")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    paths = plot_run(args.run_dir, args.out_dir)
    print("\n".join(paths) if paths else f"no plottable artifacts in {args.run_dir}")
