"""Train the MS-TCN temporal classifier on frozen tokenizer embeddings."""

import logging
import time

import numpy as np
import torch

from enrichment.fuse import ConcatProjectFusion
from enrichment.vision_features import extract_frame_features
from temporal_classifier.labels import align_labels_to_grid, build_vocab
from temporal_classifier.losses import mstcn_loss
from temporal_classifier.metrics import f1_at_k_corpus
from temporal_classifier.model import MSTCN
from pathlib import Path

from tokenizer.data import RATE_HZ, load_demo, resample_to_grid
from tokenizer.train import load_checkpoint
from tokenizer.windowing import demo_to_sequence

log = logging.getLogger(__name__)


def build_demo_sequence(tokenizer_model, mean, std, demo_path: str, window: int):
    channel_arrays, channel_timestamps, low_level_segments = load_demo(demo_path)
    telemetry, grid = resample_to_grid(channel_arrays, channel_timestamps)
    normed = (telemetry - mean) / std

    windows, centers = demo_to_sequence(normed, grid, window=window)
    x = torch.from_numpy(windows.astype(np.float32))
    with torch.no_grad():
        z_q, _ = tokenizer_model.encode_tokens(x)
    return z_q.numpy(), centers, low_level_segments


def _prepare_split(
    tokenizer_model,
    mean,
    std,
    demo_paths: list[str],
    window: int,
    vocab: dict[str, int],
    use_vision: bool = False,
    vision_encoder=None,
    camera_key: str = "hama1",
    device: str = "cuda",
    vision_cache_dir: str | None = None,
    vision_window_s: float | None = None,
    verbose: bool = True,
):
    sequences, vision_seqs, label_seqs = [], [], []
    for i, path in enumerate(demo_paths, start=1):
        started = time.time()
        embeddings, centers, segments = build_demo_sequence(tokenizer_model, mean, std, path, window)
        labels = align_labels_to_grid(centers, segments, vocab)
        sequences.append(embeddings)
        label_seqs.append(labels)
        if use_vision:
            vision_seqs.append(
                extract_frame_features(
                    path, centers, vision_encoder, camera_key, device=device,
                    cache_dir=vision_cache_dir, window_s=vision_window_s,
                )
            )
            if verbose:
                log.info(
                    f"    vision {i}/{len(demo_paths)}  {Path(path).stem}  "
                    f"{len(centers)} tokens  {time.time() - started:.1f}s"
                )
    return sequences, vision_seqs, label_seqs


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
    use_vision: bool = False,
    vision_encoder=None,
    camera_key: str = "hama1",
    fusion_out_dim: int = 128,
    vision_cache_dir: str | None = None,
    pool_vision_over_window: bool = False,
    smoothing_weight: float = 0.15,
    tau: float = 4.0,
    eval_every: int = 5,
    verbose: bool = True,
) -> dict:
    if device == "cpu":
        # See tokenizer/train.py: default CPU intra-op thread pool causes ~140x overhead on
        # per-demo batches this small.
        torch.set_num_threads(1)

    tokenizer_model, mean, std, config = load_checkpoint(tokenizer_ckpt_path)
    window = config["window"]
    # Pool over exactly the token's own duration, so the vision window and the motion window
    # describe the same slice of time by construction.
    vision_window_s = window / RATE_HZ if pool_vision_over_window else None

    if vocab is None:
        all_segments = []
        for path in train_paths + val_paths:
            _, _, segments = load_demo(path)
            all_segments.append(segments)
        vocab = build_vocab(all_segments)

    train_seqs, train_vision, train_labels = _prepare_split(
        tokenizer_model, mean, std, train_paths, window, vocab, use_vision, vision_encoder,
        camera_key, device, vision_cache_dir, vision_window_s, verbose,
    )
    val_seqs, val_vision, val_labels = _prepare_split(
        tokenizer_model, mean, std, val_paths, window, vocab, use_vision, vision_encoder,
        camera_key, device, vision_cache_dir, vision_window_s, verbose,
    )

    fusion = None
    classifier_in_dim = config["latent_dim"]
    if use_vision:
        from enrichment.vision_features import VISION_FEATURE_DIM

        fusion = ConcatProjectFusion(config["latent_dim"], VISION_FEATURE_DIM, fusion_out_dim).to(device)
        # Only the vision arm's classifier is resized; the telemetry-only baseline stays on
        # latent_dim, unchanged, since it is the control arm of the vision ablation.
        classifier_in_dim = fusion.out_dim

    model = MSTCN(classifier_in_dim, len(vocab), channels, num_layers, num_stages).to(device)
    params = list(model.parameters()) + (list(fusion.parameters()) if fusion else [])
    optimizer = torch.optim.Adam(params, lr=lr)

    def build_input(embeddings, vision_feats):
        x = torch.from_numpy(embeddings).to(device)
        if fusion is not None:
            v = torch.from_numpy(vision_feats.astype(np.float32)).to(device)
            x = fusion(x, v)
        return x.unsqueeze(0)

    def score_validation():
        model.eval()
        pairs = []
        with torch.no_grad():
            for j, labels in enumerate(val_labels):
                x = build_input(val_seqs[j], val_vision[j] if use_vision else None)
                pred = model(x)[-1].argmax(dim=1).squeeze(0).cpu().numpy()
                pairs.append((pred, labels))
        model.train()
        return f1_at_k_corpus(pairs)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        started, running = time.time(), 0.0
        for i, labels in enumerate(train_labels):
            x = build_input(train_seqs[i], train_vision[i] if use_vision else None)
            y = torch.from_numpy(labels).unsqueeze(0).to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = mstcn_loss(outputs, y, smoothing_weight, tau)
            loss.backward()
            optimizer.step()
            running += loss.item()

        train_loss = running / max(len(train_labels), 1)
        val_f1 = score_validation() if (eval_every and epoch % eval_every == 0) else None
        history.append({"epoch": epoch, "train_loss": train_loss, "val_f1": val_f1})
        if verbose:
            f1_str = f"  val_f1 {val_f1:.4f}" if val_f1 is not None else ""
            log.info(
                f"    epoch {epoch:>3}/{epochs}  loss {train_loss:.5f}{f1_str}"
                f"  ({time.time() - started:.1f}s)"
            )

    model.eval()
    pred_gt_pairs = []
    with torch.no_grad():
        for i, labels in enumerate(val_labels):
            x = build_input(val_seqs[i], val_vision[i] if use_vision else None)
            pred = model(x)[-1].argmax(dim=1).squeeze(0).cpu().numpy()
            pred_gt_pairs.append((pred, labels))

    val_f1 = f1_at_k_corpus(pred_gt_pairs)
    if history and history[-1]["val_f1"] is None:
        history[-1]["val_f1"] = val_f1
    return {"model": model, "fusion": fusion, "vocab": vocab, "val_f1": val_f1, "history": history}
