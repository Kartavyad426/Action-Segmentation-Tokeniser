import torch

from tokenizer.model import MotionTokenizer


def test_forward_shapes():
    model = MotionTokenizer(in_channels=22, window=15, latent_dim=32, num_codes=64, hidden=32)
    x = torch.randn(4, 15, 22)

    out = model(x)

    assert out["recon"].shape == (4, 15, 22)
    assert out["indices"].shape == (4,)
    assert out["loss"].ndim == 0
    assert out["recon_loss"].ndim == 0
    assert out["vq_loss"].ndim == 0


def test_encode_tokens_matches_forward_indices():
    model = MotionTokenizer(in_channels=5, window=10, latent_dim=8, num_codes=16, hidden=16)
    x = torch.randn(3, 10, 5)

    out = model(x)
    z_q, indices = model.encode_tokens(x)

    assert z_q.shape == (3, 8)
    assert torch.equal(indices, out["indices"])


def test_loss_decreases_when_overfitting_a_single_batch():
    torch.manual_seed(0)
    model = MotionTokenizer(in_channels=4, window=10, latent_dim=8, num_codes=16, hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(8, 10, 4)

    first_loss = None
    last_loss = None
    for step in range(50):
        optimizer.zero_grad()
        out = model(x)
        out["loss"].backward()
        optimizer.step()
        if step == 0:
            first_loss = out["loss"].item()
        last_loss = out["loss"].item()

    assert last_loss < first_loss
