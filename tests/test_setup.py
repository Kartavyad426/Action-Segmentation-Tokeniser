"""Sanity check that the environment is wired up correctly."""


def test_torch_cuda_available():
    import torch

    assert torch.cuda.is_available()
