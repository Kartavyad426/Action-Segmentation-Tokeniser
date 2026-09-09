"""Train the VQ-VAE motion tokenizer on a list of REASSEMBLE demo files."""

import torch
from torch.utils.data import DataLoader

from tokenizer.data import load_demo, resample_to_grid
from tokenizer.model import MotionTokenizer
from tokenizer.windowing import WindowedTelemetryDataset


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
        if verbose:
            val_str = f"  val {val_loss:.5f}" if val_loss is not None else ""
            print(f"  epoch {epoch:>3}/{epochs}  train {train_loss:.5f}{val_str}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "mean": dataset.mean,
            "std": dataset.std,
            "config": {
                "in_channels": in_channels,
                "window": window,
                "latent_dim": latent_dim,
                "num_codes": num_codes,
                "hidden": hidden,
            },
        },
        out_path,
    )
    return {"final_loss": final_loss, "history": history}


def load_checkpoint(path: str, device: str = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = MotionTokenizer(
        config["in_channels"], config["window"], config["latent_dim"], config["num_codes"], config["hidden"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint["mean"], checkpoint["std"], config
