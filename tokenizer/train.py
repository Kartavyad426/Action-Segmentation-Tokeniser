"""Train the VQ-VAE motion tokenizer on a list of REASSEMBLE demo files."""

import hashlib
import json
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tokenizer.data import load_demo, resample_to_grid
from tokenizer.model import MotionTokenizer
from tokenizer.windowing import WindowedTelemetryDataset

log = logging.getLogger(__name__)


def tokenizer_fingerprint(demo_paths, **hyperparameters) -> str:
    """Identifies the exact training configuration a checkpoint came from.

    A checkpoint carries a codebook and normalization statistics fitted to one specific set
    of demos under one specific set of hyperparameters. Reloading it for a run that changed
    either would silently apply the wrong tokenizer, and nothing downstream would notice.
    Demo order does not matter; demo membership does.
    """
    payload = {
        "demos": sorted(Path(p).stem for p in demo_paths),
        "hyperparameters": {k: hyperparameters[k] for k in sorted(hyperparameters)},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _load_all_telemetry(demo_paths: list[str]):
    telemetry_list = []
    for path in demo_paths:
        channel_arrays, channel_timestamps, _ = load_demo(path)
        telemetry, _ = resample_to_grid(channel_arrays, channel_timestamps)
        telemetry_list.append(telemetry)
    return telemetry_list


@torch.no_grad()
def _evaluate_loss(model, loader, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        total += model(batch)["loss"].item() * batch.shape[0]
        count += batch.shape[0]
    model.train()
    return total / max(count, 1)


def train_tokenizer(
    demo_paths: list[str],
    out_path: str,
    epochs: int = 20,
    window: int = 15,
    stride: int = 5,
    latent_dim: int = 32,
    num_codes: int = 512,
    hidden: int = 64,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda",
    val_paths: list[str] | None = None,
    verbose: bool = True,
    on_epoch=None,
) -> dict:
    if device == "cpu":
        # PyTorch's default CPU intra-op thread pool causes ~140x overhead on batches this
        # small (measured: 249ms/batch at default thread count vs 1.79ms/batch at 1 thread) —
        # the synchronization cost dominates for a model/batch this tiny.
        torch.set_num_threads(1)

    telemetry_list = _load_all_telemetry(demo_paths)
    in_channels = telemetry_list[0].shape[1]

    dataset = WindowedTelemetryDataset(telemetry_list, window=window, stride=stride)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Held-out windows are normalized with the *training* mean/std. Recomputing them over the
    # validation demos would leak their distribution into the evaluation and make the two
    # losses incomparable.
    val_loader = None
    if val_paths:
        val_dataset = WindowedTelemetryDataset(
            _load_all_telemetry(val_paths), window=window, stride=stride,
            mean=dataset.mean, std=dataset.std,
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = MotionTokenizer(in_channels, window, latent_dim, num_codes, hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def save(path: str, epoch: int, train_loss: float, val_loss):
        """Written every epoch, atomically. A hang or a kill at epoch 17 of 20 must not
        throw away the whole run -- the previous 16 epochs are still a usable tokenizer."""
        payload = {
            "model_state": model.state_dict(),
            "mean": dataset.mean,
            "std": dataset.std,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": {
                "fingerprint": tokenizer_fingerprint(
                    demo_paths, window=window, stride=stride, latent_dim=latent_dim,
                    num_codes=num_codes, hidden=hidden, epochs=epochs, batch_size=batch_size, lr=lr,
                ),
                "in_channels": in_channels,
                "window": window,
                "latent_dim": latent_dim,
                "num_codes": num_codes,
                "hidden": hidden,
            },
        }
        torch.save(payload, path + ".tmp")
        os.replace(path + ".tmp", path)

    best_path = out_path[:-3] + ".best.pt" if out_path.endswith(".pt") else out_path + ".best"
    best_val_loss, best_epoch = None, None
    final_loss = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            out["loss"].backward()
            optimizer.step()
            final_loss = out["loss"].item()
            running += final_loss * batch.shape[0]
            seen += batch.shape[0]

        train_loss = running / max(seen, 1)
        val_loss = _evaluate_loss(model, val_loader, device) if val_loader is not None else None
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        save(out_path, epoch, train_loss, val_loss)
        if val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
            best_val_loss, best_epoch = val_loss, epoch
            save(best_path, epoch, train_loss, val_loss)
        if on_epoch is not None:
            on_epoch(epoch, history)
        if verbose:
            val_str = f"  val {val_loss:.5f}" if val_loss is not None else ""
            log.info(f"  tokenizer epoch {epoch:>3}/{epochs}  train {train_loss:.5f}{val_str}")

    return {
        "final_loss": final_loss,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_checkpoint": best_path if best_epoch is not None else None,
    }


def load_checkpoint(path: str, device: str = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = MotionTokenizer(
        config["in_channels"], config["window"], config["latent_dim"], config["num_codes"], config["hidden"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint["mean"], checkpoint["std"], config
