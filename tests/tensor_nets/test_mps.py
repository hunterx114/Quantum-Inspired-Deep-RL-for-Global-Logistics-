"""Numerical, gradient, and parameter-scaling tests for MPS layers."""

from __future__ import annotations

import torch
from torch import nn

from src.tensor_nets import MPSEncoder, MPSLinear


def test_svd_factorization_matches_dense_linear_at_sufficient_bond_dimension() -> None:
    torch.manual_seed(31)
    dense = nn.Linear(8, 12)
    factorized = MPSLinear.from_dense(
        in_shape=(2, 2, 2),
        out_shape=(2, 3, 2),
        bond_dim=32,
        weight=dense.weight,
        bias=dense.bias,
    )
    inputs = torch.randn(7, 8)

    torch.testing.assert_close(factorized(inputs), dense(inputs), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(factorized.to_dense(), dense.weight, rtol=1e-5, atol=1e-6)


def test_mps_linear_preserves_batch_modes_and_propagates_gradients() -> None:
    layer = MPSLinear((2, 3), (2, 2), bond_dim=3)
    inputs = torch.randn(5, 2, 3, requires_grad=True)
    output = layer(inputs)
    assert output.shape == (5, 4)

    output.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert all(core.grad is not None for core in layer.cores)
    assert all(torch.isfinite(core.grad).all() for core in layer.cores)


def test_mps_parameter_count_is_compressed_against_dense_linear() -> None:
    layer = MPSLinear(64, 64, bond_dim=4)
    dense_parameter_count = 64 * 64 + 64
    assert sum(parameter.numel() for parameter in layer.parameters()) < dense_parameter_count


def test_mps_encoder_returns_compact_embedding_with_gradients() -> None:
    encoder = MPSEncoder(input_dim=16, embedding_dim=6, bond_dim=3, n_modes=3)
    states = torch.randn(4, 16, requires_grad=True)
    embeddings = encoder(states)
    assert embeddings.shape == (4, 6)
    embeddings.sum().backward()
    assert states.grad is not None
    assert torch.isfinite(states.grad).all()


def test_svd_truncation_respects_the_requested_bond_dimension() -> None:
    dense = nn.Linear(64, 64, bias=False)
    layer = MPSLinear.from_dense(
        64,
        64,
        bond_dim=1,
        weight=dense.weight,
        truncate=True,
    )
    assert max(core.shape[-1] for core in layer.cores[:-1]) == 1
