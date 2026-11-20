# MPS / tensor-train layers

## What this implementation represents

`MPSLinear` represents a linear weight matrix as a sequence of four-index
cores, each with a left bond, an input mode, an output mode, and a right bond.
The map is evaluated by contracting the input with those cores using native
PyTorch `einsum`; inference does not first materialize the dense matrix.
`to_dense()` is available for inspection and verification. `from_dense()` uses
TT-SVD and can cap intermediate ranks with `bond_dim`.

For integer feature sizes, the layer balances their factors across two modes by
default; explicit mode-shape tuples or `n_modes` provide control of the
factorization. `MPSEncoder` applies a bounded MPS projection to form a compact
state embedding.

## Parameter and compute scaling

For `N` paired input/output modes with physical dimensions `d_in[k]` and
`d_out[k]`, core `k` contains
`r[k] * d_in[k] * d_out[k] * r[k+1]` parameters. With roughly uniform physical
dimension `d` and bond dimension `chi`, parameter count is approximately
`O(N d^2 chi^2)`, versus `O(d^(2N))` for an unrestricted dense map over the
same paired modes. Contracting a vector uses work proportional to the core
sizes, approximately `O(N d^2 chi^2)` for bounded bonds.

For a general `N`-mode state tensor with local dimension `d`, an MPS stores
`O(N d chi^2)` parameters rather than `O(d^N)` dense state coefficients. A
linear operator has both input and output physical indices, hence the extra
physical factor above. These savings are not guaranteed to preserve accuracy:
bond dimension caps the rank at every cut, and a small bond can discard
important correlations.

## Scope and scientific caveat

This is a **quantum-inspired classical simulation** using ordinary PyTorch
tensors, autograd, and CPU/GPU operations. It does not execute on quantum
hardware, and parameter compression alone does not establish a performance or
generalization advantage. The included dense and tensor-network benchmarks
must be interpreted using measured runs and matched configurations.
