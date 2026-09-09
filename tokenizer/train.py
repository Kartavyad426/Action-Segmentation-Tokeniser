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

    model = MotionTokenizer(in_channels, window, latent_dim, num_codes, hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    final_loss = None
    for _ in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            out["loss"].backward()
            optimizer.step()
            final_loss = out["loss"].item()

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
    return {"final_loss": final_loss}


def load_checkpoint(path: str, device: str = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = MotionTokenizer(
        config["in_channels"], config["window"], config["latent_dim"], config["num_codes"], config["hidden"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint["mean"], checkpoint["std"], config
