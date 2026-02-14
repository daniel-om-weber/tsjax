"""S5 (Simplified State Space Sequence) model using Flax NNX.

Ported from the canonical linen implementation at lindermanlab/S5.
Uses diagonal state spaces with HiPPO initialization and jax.lax.associative_scan
for parallelizable training.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import lecun_normal
from jax.numpy.linalg import eigh

# ---------------------------------------------------------------------------
# HiPPO initialization (private)
# ---------------------------------------------------------------------------


def _make_hippo(N):
    P = jnp.sqrt(1 + 2 * jnp.arange(N))
    A = P[:, None] * P[None, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N, dtype=jnp.float32))
    return -A


def _make_nplr_hippo(N):
    hippo = _make_hippo(N)
    P = jnp.sqrt(jnp.arange(N) + 0.5)
    B = jnp.sqrt(2 * jnp.arange(N) + 1.0)
    return hippo, P, B


def _make_dplr_hippo(N):
    A, P, B = _make_nplr_hippo(N)
    S = A + P[:, None] * P[None, :]
    S_diag = jnp.diagonal(S)
    Lambda_real = jnp.mean(S_diag) * jnp.ones_like(S_diag)
    Lambda_imag, V = eigh(S * -1j)
    Vinv = V.conj().T
    return Lambda_real + 1j * Lambda_imag, V, Vinv


# ---------------------------------------------------------------------------
# Discretization and parallel scan (private)
# ---------------------------------------------------------------------------


def _discretize_zoh(Lambda, B_tilde, Delta):
    Identity = jnp.ones(Lambda.shape[0])
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = (1.0 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


def _binary_operator(q_i, q_j):
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def _apply_ssm(Lambda_bar, B_bar, C_tilde, D, input_sequence):
    Bu_elements = input_sequence @ B_bar.T  # (*batch, L, H) @ (H, P) -> (*batch, L, P)
    Lambda_elements = jnp.broadcast_to(Lambda_bar, Bu_elements.shape)
    _, xs = jax.lax.associative_scan(_binary_operator, (Lambda_elements, Bu_elements), axis=-2)
    ys = 2 * (xs @ C_tilde.T).real  # (*batch, L, P) @ (P, H) -> (*batch, L, H)
    return ys + input_sequence * D


# ---------------------------------------------------------------------------
# Parameter initialization helpers (private)
# ---------------------------------------------------------------------------


def _init_VinvB(key, shape, Vinv):
    B = lecun_normal()(key, shape)
    VinvB = Vinv @ B
    return jnp.concatenate((VinvB.real[..., None], VinvB.imag[..., None]), axis=-1)


def _init_CV(key, H, P, V):
    k1, k2 = jax.random.split(key)
    C_complex = lecun_normal()(k1, (H, P)) + 1j * lecun_normal()(k2, (H, P))
    CV = C_complex @ V
    return CV.real, CV.imag


def _init_log_steps(key, P, dt_min=0.001, dt_max=0.1):
    return jax.random.uniform(key, (P,)) * (jnp.log(dt_max) - jnp.log(dt_min)) + jnp.log(dt_min)


# ---------------------------------------------------------------------------
# S5 layer (private)
# ---------------------------------------------------------------------------


class _S5Layer(nnx.Module):
    """Single S5 layer: SSM + GELU + skip connection + LayerNorm."""

    def __init__(self, d_model: int, hippo_params, *, rngs: nnx.Rngs):
        Lambda_init, V, Vinv = hippo_params
        P = Lambda_init.shape[0]
        H = d_model

        # SSM eigenvalues: real part stored as log(-re) to guarantee stability
        self.log_lambda_re = nnx.Param(jnp.log(-Lambda_init.real))
        self.lambda_im = nnx.Param(Lambda_init.imag)

        # Input matrix B_tilde = V^{-1} B, stored as (P, H, 2)
        key = rngs.params()
        B_param = _init_VinvB(key, (P, H), Vinv)
        self.B_re = nnx.Param(B_param[..., 0])
        self.B_im = nnx.Param(B_param[..., 1])

        # Output matrix C_tilde = C V, stored as separate (H, P) real/imag
        key = rngs.params()
        C_re, C_im = _init_CV(key, H, P, V)
        self.C_re = nnx.Param(C_re)
        self.C_im = nnx.Param(C_im)

        # Feedthrough
        key = rngs.params()
        self.D = nnx.Param(jax.random.normal(key, (H,)))

        # Learnable log time step
        key = rngs.params()
        self.log_step = nnx.Param(_init_log_steps(key, P))

        # Post-SSM normalization
        self.norm = nnx.LayerNorm(num_features=H, rngs=rngs)

    def __call__(self, x):
        # x: (*batch, seq_len, d_model)
        Lambda = -jnp.exp(self.log_lambda_re[...]) + 1j * self.lambda_im[...]
        B_tilde = self.B_re[...] + 1j * self.B_im[...]
        C_tilde = self.C_re[...] + 1j * self.C_im[...]

        step = jnp.exp(self.log_step[...])
        Lambda_bar, B_bar = _discretize_zoh(Lambda, B_tilde, step)

        y = _apply_ssm(Lambda_bar, B_bar, C_tilde, self.D[...], x)
        y = jax.nn.gelu(y)
        return self.norm(x + y)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class S5(nnx.Module):
    """Multi-layer S5 state space model.

    Args:
        input_size: number of input features.
        output_size: number of output features.
        hidden_size: internal feature dimension (d_model).
        state_size: SSM state dimension (P). Must be even (conjugate symmetry).
        num_layers: number of stacked S5 layers.
        rngs: Flax NNX RNG container.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        state_size: int = 64,
        num_layers: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        P = state_size // 2  # conjugate symmetry halves the state
        hippo_params = _make_dplr_hippo(P)

        self.input_proj = nnx.Linear(in_features=input_size, out_features=hidden_size, rngs=rngs)
        self.layers = nnx.List(
            [_S5Layer(hidden_size, hippo_params, rngs=rngs) for _ in range(num_layers)]
        )
        self.output_proj = nnx.Linear(in_features=hidden_size, out_features=output_size, rngs=rngs)

    def __call__(self, u):
        """Forward pass.

        u: ``(*batch, seq_len, input_size)`` -> ``(*batch, seq_len, output_size)``
        """
        x = self.input_proj(u)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)
