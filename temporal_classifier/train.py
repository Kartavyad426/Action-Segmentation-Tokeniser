"""Train the MS-TCN temporal classifier on frozen tokenizer embeddings."""

import numpy as np
import torch
import torch.nn as nn

from temporal_classifier.labels import align_labels_to_grid, build_vocab
from temporal_classifier.metrics import f1_at_k
from temporal_classifier.model import MSTCN
from tokenizer.data import load_demo, resample_to_grid
from tokenizer.train import load_checkpoint
from tokenizer.windowing import demo_to_sequence


def build_demo_sequence(tokenizer_model, mean, std, demo_path: str, window: int):
    channel_arrays, channel_timestamps, low_level_segments = load_demo(demo_path)
    telemetry, grid = resample_to_grid(channel_arrays, channel_timestamps)
    normed = (telemetry - mean) / std

    windows, centers = demo_to_sequence(normed, grid, window=window)
    x = torch.from_numpy(windows.astype(np.float32))
    with torch.no_grad():
        z_q, _ = tokenizer_model.encode_tokens(x)
    return z_q.numpy(), centers, low_level_segments


def _prepare_split(tokenizer_model, mean, std, demo_paths: list[str], window: int, vocab: dict[str, int]):
    sequences, label_seqs = [], []
    for path in demo_paths:
        embeddings, centers, segments = build_demo_sequence(tokenizer_model, mean, std, path, window)
        labels = align_labels_to_grid(centers, segments, vocab)
        sequences.append(embeddings)
        label_seqs.append(labels)
    return sequences, label_seqs


def run_training(
    tokenizer_ckpt_path: str,
    train_paths: list[str],
    val_paths: list[str],
    vocab: dict[str, int] | None = None,
    epochs: int = 30,
    channels: int = 64,
    num_layers: int = 9,
    num_stages: int = 3,
    lr: float = 1e-3,
    device: str = "cuda",
) -> dict:
    if device == "cpu":
        # See tokenizer/train.py: default CPU intra-op thread pool causes ~140x overhead on
        # per-demo batches this small.
        torch.set_num_threads(1)

    tokenizer_model, mean, std, config = load_checkpoint(tokenizer_ckpt_path)
    window = config["window"]

    if vocab is None:
        all_segments = []
        for path in train_paths + val_paths:
            _, _, segments = load_demo(path)
            all_segments.append(segments)
        vocab = build_vocab(all_segments)

    train_seqs, train_labels = _prepare_split(tokenizer_model, mean, std, train_paths, window, vocab)
    val_seqs, val_labels = _prepare_split(tokenizer_model, mean, std, val_paths, window, vocab)

    model = MSTCN(config["latent_dim"], len(vocab), channels, num_layers, num_stages).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for embeddings, labels in zip(train_seqs, train_labels):
            x = torch.from_numpy(embeddings).unsqueeze(0).to(device)
            y = torch.from_numpy(labels).unsqueeze(0).to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = sum(loss_fn(out, y) for out in outputs)
            loss.backward()
            optimizer.step()

    model.eval()
    f1_scores = []
    with torch.no_grad():
        for embeddings, labels in zip(val_seqs, val_labels):
            x = torch.from_numpy(embeddings).unsqueeze(0).to(device)
            pred = model(x)[-1].argmax(dim=1).squeeze(0).cpu().numpy()
            f1_scores.append(f1_at_k(pred, labels))

    val_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return {"model": model, "vocab": vocab, "val_f1": val_f1}
