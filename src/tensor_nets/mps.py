"""Matrix Product State (tensor-train) layers and dense-weight factorization."""

from __future__ import annotations

from collections.abc import Sequence
import string

import torch
from torch import nn


_EINSUM_LABELS = string.ascii_lowercase + string.ascii_uppercase


def _factor_modes(size: int, n_modes: int) -> tuple[int, ...]:
    """Split an integer dimension into balanced integer factors."""
    if size < 1:
        raise ValueError("all dimensions must be positive")
    factors: list[int] = []
    remainder = size
    divisor = 2
    while divisor * divisor <= remainder:
        while remainder % divisor == 0:
            factors.append(divisor)
            remainder //= divisor
        divisor += 1
    if remainder > 1:
        factors.append(remainder)
    modes = [1] * n_modes
    for factor in sorted(factors, reverse=True):
        smallest = min(range(n_modes), key=modes.__getitem__)
        modes[smallest] *= factor
    return tuple(modes)


def _normalize_shape(
    shape: int | Sequence[int],
    n_modes: int,
) -> tuple[int, ...]:
    if isinstance(shape, int):
        return _factor_modes(shape, n_modes)
    normalized = tuple(int(dimension) for dimension in shape)
    if not normalized or any(dimension < 1 for dimension in normalized):
        raise ValueError("mode shapes must contain positive dimensions")
    if len(normalized) != n_modes:
        raise ValueError("input and output mode shapes must have the same length")
    return normalized


def _shape_product(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


class MPSLinear(nn.Module):
    """Linear map stored as a train of four-index tensor cores.

    The weight convention matches ``nn.Linear``: ``to_dense()`` returns
    ``(out_features, in_features)`` and the forward result is flattened on its
    final feature axis. Integer feature sizes are balanced into two modes by
    default; explicit tuples specify the input/output mode dimensions.
    """

    def __init__(
        self,
        in_shape: int | Sequence[int],
        out_shape: int | Sequence[int],
        bond_dim: int,
        bias: bool = True,
        n_modes: int | None = None,
    ) -> None:
        super().__init__()
        if bond_dim < 1:
            raise ValueError("bond_dim must be positive")
        if n_modes is None:
            tuple_lengths = [
                len(shape)
                for shape in (in_shape, out_shape)
                if not isinstance(shape, int)
            ]
            if tuple_lengths and len(set(tuple_lengths)) != 1:
                raise ValueError("input and output mode shapes must have equal length")
            n_modes = tuple_lengths[0] if tuple_lengths else 2
        if not 1 <= n_modes <= 16:
            raise ValueError("n_modes must be between 1 and 16 for einsum contraction")
        self.in_modes = _normalize_shape(in_shape, n_modes)
        self.out_modes = _normalize_shape(out_shape, n_modes)
        self.in_features = _shape_product(self.in_modes)
        self.out_features = _shape_product(self.out_modes)
        self.bond_dim = bond_dim

        paired_dimensions = [
            input_mode * output_mode
            for input_mode, output_mode in zip(self.in_modes, self.out_modes)
        ]
        ranks = [1]
        for cut in range(1, n_modes):
            left_dimension = _shape_product(paired_dimensions[:cut])
            right_dimension = _shape_product(paired_dimensions[cut:])
            ranks.append(min(bond_dim, left_dimension, right_dimension))
        ranks.append(1)
        self.cores = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        ranks[index],
                        self.in_modes[index],
                        self.out_modes[index],
                        ranks[index + 1],
                    )
                )
                for index in range(n_modes)
            ]
        )
        self.bias = (
            nn.Parameter(torch.empty(self.out_features)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize core products to a stable, dense-linear scale."""
        internal_rank_product = 1
        for core in self.cores[:-1]:
            internal_rank_product *= core.shape[-1]
        standard_deviation = (
            1.0 / (self.in_features * internal_rank_product)
        ) ** (1.0 / (2 * len(self.cores)))
        for core in self.cores:
            nn.init.normal_(core, mean=0.0, std=standard_deviation)
        if self.bias is not None:
            bound = self.in_features**-0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Contract the input and cores without materializing the dense matrix."""
        if inputs.ndim < 1:
            raise ValueError("inputs must have a feature dimension")
        if tuple(inputs.shape[-len(self.in_modes) :]) == self.in_modes:
            leading_shape = inputs.shape[: -len(self.in_modes)]
        elif inputs.shape[-1] == self.in_features:
            leading_shape = inputs.shape[:-1]
        else:
            raise ValueError(
                f"inputs must end with shape {self.in_modes} or "
                f"feature size {self.in_features}"
            )
        batch_size = _shape_product(leading_shape) if leading_shape else 1
        state = inputs.reshape(batch_size, *self.in_modes)

        batch_label = _EINSUM_LABELS[0]
        input_labels = _EINSUM_LABELS[1 : 1 + len(self.cores)]
        output_start = 1 + len(self.cores)
        output_labels = _EINSUM_LABELS[output_start : output_start + len(self.cores)]
        rank_start = output_start + len(self.cores)
        rank_labels = _EINSUM_LABELS[rank_start : rank_start + len(self.cores) + 1]
        if len(rank_labels) != len(self.cores) + 1:
            raise ValueError("too many MPS cores for a distinct-index contraction")

        state_labels = batch_label + "".join(input_labels)
        for index, core in enumerate(self.cores):
            if index == 0:
                core_labels = (
                    input_labels[index]
                    + output_labels[index]
                    + rank_labels[index + 1]
                )
            else:
                core_labels = (
                    rank_labels[index]
                    + input_labels[index]
                    + output_labels[index]
                    + rank_labels[index + 1]
                )
            next_state_labels = (
                batch_label
                + "".join(input_labels[index + 1 :])
                + "".join(output_labels[: index + 1])
                + rank_labels[index + 1]
            )
            state = torch.einsum(
                f"{state_labels},{core_labels}->{next_state_labels}",
                state,
                core[0] if index == 0 else core,
            )
            state_labels = next_state_labels

        output = state.squeeze(-1).reshape(*leading_shape, self.out_features)
        if self.bias is not None:
            output = output + self.bias
        return output

    def to_dense(self) -> torch.Tensor:
        """Contract all cores and return the equivalent dense linear weight."""
        contracted = self.cores[0][0]
        for core in self.cores[1:]:
            contracted = torch.einsum("...r,rios->...ios", contracted, core)
        contracted = contracted.squeeze(-1)
        n_modes = len(self.cores)
        permutation = list(range(0, 2 * n_modes, 2)) + list(
            range(1, 2 * n_modes, 2)
        )
        matrix = contracted.permute(permutation).reshape(
            self.in_features, self.out_features
        )
        return matrix.transpose(0, 1).contiguous()

    @classmethod
    def from_dense(
        cls,
        in_shape: int | Sequence[int],
        out_shape: int | Sequence[int],
        bond_dim: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        n_modes: int | None = None,
        truncate: bool = True,
    ) -> "MPSLinear":
        """Factor a dense ``nn.Linear`` weight with optional SVD rank truncation."""
        if not weight.is_floating_point():
            raise ValueError("weight must have a floating-point dtype")
        if n_modes is None:
            tuple_lengths = [
                len(shape)
                for shape in (in_shape, out_shape)
                if not isinstance(shape, int)
            ]
            n_modes = tuple_lengths[0] if tuple_lengths else 2
        in_modes = _normalize_shape(in_shape, n_modes)
        out_modes = _normalize_shape(out_shape, n_modes)
        in_features = _shape_product(in_modes)
        out_features = _shape_product(out_modes)
        if tuple(weight.shape) != (out_features, in_features):
            raise ValueError(
                f"weight must have shape {(out_features, in_features)}, "
                f"got {tuple(weight.shape)}"
            )
        if bias is not None and tuple(bias.shape) != (out_features,):
            raise ValueError(f"bias must have shape {(out_features,)}")
        if bond_dim < 1:
            raise ValueError("bond_dim must be positive")

        layer = cls(
            in_modes,
            out_modes,
            bond_dim,
            bias=bias is not None,
            n_modes=n_modes,
        ).to(device=weight.device, dtype=weight.dtype)
        if n_modes == 1:
            cores = [weight.transpose(0, 1).reshape(1, in_modes[0], out_modes[0], 1)]
        else:
            tensor = weight.transpose(0, 1).reshape(*in_modes, *out_modes)
            pair_order = [
                axis
                for mode in range(n_modes)
                for axis in (mode, n_modes + mode)
            ]
            paired_dimensions = [
                in_modes[index] * out_modes[index] for index in range(n_modes)
            ]
            remainder = tensor.permute(pair_order).reshape(
                1, *paired_dimensions
            )
            left_rank = 1
            cores = []
            for index in range(n_modes - 1):
                matrix = remainder.reshape(
                    left_rank * paired_dimensions[index], -1
                )
                left_vectors, singular_values, right_vectors = torch.linalg.svd(
                    matrix, full_matrices=False
                )
                rank = singular_values.numel()
                if truncate:
                    rank = min(rank, bond_dim)
                core = left_vectors[:, :rank].reshape(
                    left_rank,
                    in_modes[index],
                    out_modes[index],
                    rank,
                )
                cores.append(core)
                remainder = (
                    singular_values[:rank, None] * right_vectors[:rank]
                ).reshape(rank, *paired_dimensions[index + 1 :])
                left_rank = rank
            cores.append(
                remainder.reshape(
                    left_rank, in_modes[-1], out_modes[-1], 1
                )
            )
        layer.cores = nn.ParameterList(
            [nn.Parameter(core.detach().contiguous()) for core in cores]
        )
        if bias is not None:
            assert layer.bias is not None
            with torch.no_grad():
                layer.bias.copy_(bias)
        return layer


class MPSEncoder(nn.Module):
    """Compress flat state features into a compact MPS embedding."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        bond_dim: int,
        n_modes: int = 4,
    ) -> None:
        super().__init__()
        if input_dim < 1 or embedding_dim < 1:
            raise ValueError("input_dim and embedding_dim must be positive")
        self.linear = MPSLinear(
            input_dim,
            embedding_dim,
            bond_dim,
            n_modes=n_modes,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return a bounded compact representation of the state."""
        return torch.tanh(self.linear(states))
