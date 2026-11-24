"""
Zero-shot RL agents: TDJepa.

Implements forward-backward style agents with JEPA predictors and encoders,
built on AMAGO's BaseAgent with shared sequence backbone and multi-gamma updates.
"""

from __future__ import annotations

import functools
import itertools
import math
import time
from typing import Optional, Type, Tuple, Any, Iterable, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
import gin
import gymnasium as gym

from amago import utils
from amago.loading import Batch, MAGIC_PAD_VAL
import abc

from amago.agent import (
    BaseAgent,
    Multigammas,
    binary_filter,
    register_agent,
    nstep_return,
)
from amago.nets import TstepEncoder, TrajEncoder
from amago.nets.actor_critic import (
    _EinMixEnsemble,
    gammas_as_input_seq,
    BaseActorHead,
    BaseCriticHead,
    NCritics,
    NCriticsTwoHot,
    PopArtLayer,
    Actor,
)
from amago.nets.ff import Normalization
from amago.nets.utils import activation_switch, add_activation_log
from einops.layers.torch import EinMix as Mix

from metabfm.rl.policy_tstep import PolicyTstepEncoder
from metabfm.rl.dynamics_learner import sanitize_published
from metabfm.rl.conditioning import fourier_encode

NO_REWARD: float = -128.0


def pairwise_cosine(z: torch.Tensor) -> float:
    """Mean off-diagonal cosine similarity among a batch of vectors."""
    z_normed = F.normalize(z, dim=-1)
    sim = z_normed @ z_normed.T
    n = z.shape[0]
    if n < 2:
        return 0.0
    mask = ~torch.eye(n, device=z.device, dtype=torch.bool)
    return sim[mask].mean().item()


class FBNormalization(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return math.sqrt(self.d_model) * F.normalize(x, dim=-1)


def _polyak_ema_inplace(target: nn.Module, online: nn.Module, tau: float) -> None:
    """Fused in-place Polyak EMA: ``target := (1 - tau) * target + tau * online``.

    Iterates ``target.parameters()`` zipped with ``online.parameters()``
    (``deepcopy`` preserves registration order), including frozen
    parameters; buffers are intentionally NOT updated. Writes in place via
    ``.data``.

    Uses ``_foreach_mul_`` + ``_foreach_add_`` so the update is two kernel
    launches per ``(device, dtype)`` group rather than several per
    parameter.
    """
    target_data = [p.data for p in target.parameters()]
    online_data = [p.data for p in online.parameters()]
    if not target_data:
        return
    torch._foreach_mul_(target_data, 1.0 - tau)
    torch._foreach_add_(target_data, online_data, alpha=tau)


def _full_copy_inplace(target: nn.Module, online: nn.Module) -> None:
    """Fused in-place hard copy: ``target := online``.

    Same contract as ``_polyak_ema_inplace``: parameters only (frozen
    included, buffers untouched), written in place, one ``_foreach_copy_``
    per ``(device, dtype)`` group.
    """
    target_data = [p.data for p in target.parameters()]
    online_data = [p.data for p in online.parameters()]
    if not target_data:
        return
    torch._foreach_copy_(target_data, online_data)


class ParallelLayerNorm(nn.Module):
    """Per-ensemble-member LayerNorm for the (B, L, C, D) layout.

    Affine parameters have shape ``(1, 1, C, D)`` so they broadcast over
    parallel-ensemble ``(B, L, C, D)`` activations. ``F.layer_norm``
    normalizes over the last dim (D); per-ensemble-member ``weight`` and
    ``bias`` then scale and shift.
    """

    def __init__(self, normalized_shape: int, n_parallel: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.n_parallel = n_parallel
        self.normalized_shape = [int(normalized_shape)]
        self.weight = nn.Parameter(torch.ones(1, 1, n_parallel, normalized_shape))
        self.bias = nn.Parameter(torch.zeros(1, 1, n_parallel, normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(x, self.normalized_shape, None, None, self.eps)
        return x * self.weight + self.bias


class _EinMixTwoTower(nn.Module):
    """Two-tower predictor (TD-JEPA ``ForwardMap``-style).

    Splits inputs into two independent embedding pathways before a shared
    trunk. Each ensemble member has its own weights for both towers and the
    trunk, batched as a single ``c d_in d_out`` ``EinMix`` matmul.

    Topology with ``tower_layers=L_t``, ``trunk_layers=L_k``, residual stream
    width ``d_hidden``, ensemble size ``C``::

        Tower A:  Mix(in_a -> d_hidden)            # always; emerges C dim
                  [ParallelLayerNorm + Tanh]        # if input_pre_norm=True
                  (L_t - 2) x [Mix(d -> d) + act]   # middle blocks
                  Mix(d -> d_hidden//2) + act       # tower output, ReLU/leaky
        Tower B:  same shape, independent weights, possibly different in_b.
        Concat(A, B)  ->  (B, L, C, d_hidden)
        Trunk:    L_k x [Mix(d -> d) + act]         # trunk hidden
                  Mix(d -> out_dim)                 # final, no activation
    """

    def __init__(
        self,
        ensemble_size: int,
        inp_dim_a: int,
        inp_dim_b: int,
        d_hidden: int,
        out_dim: int,
        tower_layers: int = 2,
        trunk_layers: int = 1,
        activation: str = "leaky_relu",
        dropout_p: float = 0.0,
        input_pre_norm: bool = True,
    ):
        super().__init__()
        assert (
            tower_layers >= 2
        ), f"tower_layers must be >= 2; got {tower_layers}"
        assert (
            trunk_layers >= 1
        ), f"trunk_layers must be >= 1 (at least one trunk hidden block); got {trunk_layers}"
        assert (
            d_hidden % 2 == 0
        ), f"d_hidden must be even so each tower can output d_hidden//2; got {d_hidden}"

        self.ensemble_size = ensemble_size
        self.d_hidden = d_hidden
        self.out_dim = out_dim
        self.tower_layers = tower_layers
        self.trunk_layers = trunk_layers
        self.input_pre_norm = input_pre_norm
        self.activation = activation_switch(activation)
        self.dropout = nn.Dropout(dropout_p)

        d_tower_out = d_hidden // 2
        self._build_tower(
            "a", inp_dim_a, d_hidden, d_tower_out, tower_layers, ensemble_size
        )
        self._build_tower(
            "b", inp_dim_b, d_hidden, d_tower_out, tower_layers, ensemble_size
        )

        self.trunk_layers_list = nn.ModuleList(
            [
                Mix(
                    "b l c d_in -> b l c d_out",
                    weight_shape="c d_in d_out",
                    bias_shape="c d_out",
                    c=ensemble_size,
                    d_in=d_hidden,
                    d_out=d_hidden,
                )
                for _ in range(trunk_layers)
            ]
        )
        self.trunk_out = Mix(
            "b l c d_in -> b l c d_out",
            weight_shape="c d_in d_out",
            bias_shape="c d_out",
            c=ensemble_size,
            d_in=d_hidden,
            d_out=out_dim,
        )

    def _build_tower(
        self,
        name: str,
        inp_dim: int,
        d_hidden: int,
        d_tower_out: int,
        tower_layers: int,
        ensemble_size: int,
    ) -> None:
        first = Mix(
            "b l d_in -> b l c d_out",
            weight_shape="c d_in d_out",
            bias_shape="c d_out",
            c=ensemble_size,
            d_in=inp_dim,
            d_out=d_hidden,
        )
        norm = (
            ParallelLayerNorm(d_hidden, ensemble_size) if self.input_pre_norm else None
        )
        middles = nn.ModuleList(
            [
                Mix(
                    "b l c d_in -> b l c d_out",
                    weight_shape="c d_in d_out",
                    bias_shape="c d_out",
                    c=ensemble_size,
                    d_in=d_hidden,
                    d_out=d_hidden,
                )
                for _ in range(tower_layers - 2)
            ]
        )
        tower_out = Mix(
            "b l c d_in -> b l c d_out",
            weight_shape="c d_in d_out",
            bias_shape="c d_out",
            c=ensemble_size,
            d_in=d_hidden,
            d_out=d_tower_out,
        )
        setattr(self, f"{name}_first", first)
        if norm is not None:
            setattr(self, f"{name}_norm", norm)
        setattr(self, f"{name}_middles", middles)
        setattr(self, f"{name}_out", tower_out)

    def _run_tower(self, x: torch.Tensor, name: str) -> torch.Tensor:
        first = getattr(self, f"{name}_first")
        middles = getattr(self, f"{name}_middles")
        tower_out = getattr(self, f"{name}_out")
        h = first(x)
        if self.input_pre_norm:
            h = getattr(self, f"{name}_norm")(h)
            h = torch.tanh(h)
        else:
            h = self.activation(h)
        h = self.dropout(h)
        for layer in middles:
            h = self.dropout(self.activation(layer(h)))
        h = self.dropout(self.activation(tower_out(h)))
        return h

    def forward(
        self, x_a: torch.Tensor, x_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_a = self._run_tower(x_a, "a")
        h_b = self._run_tower(x_b, "b")
        h = torch.cat([h_a, h_b], dim=-1)
        for layer in self.trunk_layers_list:
            h = self.dropout(self.activation(layer(h)))
        out = self.trunk_out(h)
        return out, h


@gin.configurable
class JepaPredictor(nn.Module):
    """JEPA forward model prediction head.

    Args:
        two_tower: if True, replace the fused single-MLP backbone with a
            two-tower predictor (``_EinMixTwoTower``). Tower A consumes
            ``cat(state, z, gamma)``, tower B consumes
            ``cat(state, action, gamma)``; the two ``d_hidden//2``-dim
            outputs are concatenated and run through a shared trunk. Only
            supported for continuous actions; ``n_layers`` is ignored in
            this mode.
        tower_layers: per-tower depth when ``two_tower=True``. >=2.
        trunk_layers: trunk hidden depth when ``two_tower=True``. >=1.
        input_pre_norm: when ``two_tower=True``, apply ParallelLayerNorm+Tanh
            after the first Mix in each tower.
        loss_type: ``"mse"`` (default) or ``"two_hot"`` for per-coord
            categorical regression (Dreamer-V3 / "Stop Regressing" style).
            In two-hot mode the final layer widens from ``out_dim`` to
            ``out_dim * two_hot_bins`` per gamma slot, and ``forward()``
            returns the expected value under per-coord softmax.
        two_hot_min / two_hot_max: per-coord bin range; used only when
            ``loss_type="two_hot"``. Both endpoints are inclusive bin centers.
        two_hot_bins: bin count per coordinate; used only when
            ``loss_type="two_hot"``.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        z_dim: int,
        out_dim: int,
        discrete: bool,
        gammas: torch.Tensor,
        num_ensemble: int = 2,
        d_hidden: int = 1024,
        n_layers: int = 3,
        dropout_p: float = 0.0,
        activation: str = "leaky_relu",
        two_tower: bool = False,
        tower_layers: int = 2,
        trunk_layers: int = 1,
        input_pre_norm: bool = True,
        loss_type: str = "mse",
        two_hot_min: float = -33.3,
        two_hot_max: float = 33.3,
        two_hot_bins: int = 12,
    ):
        super().__init__()
        assert loss_type in (
            "mse",
            "two_hot",
        ), f"loss_type must be 'mse' or 'two_hot'; got {loss_type!r}"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.gammas = gammas
        self.num_gammas = len(self.gammas)
        self.num_ensemble = num_ensemble
        self.z_dim = z_dim
        self.out_dim = out_dim
        self.two_tower = two_tower
        self.loss_type = loss_type

        # `_fused_per_step` is the per-(K,B,L,C,G) output width of the final
        # linear: `out_dim` in MSE mode, `out_dim * num_bins` in two-hot mode.
        # All downstream shape arithmetic (discrete action-mux, einops
        # rearrange) treats this as an opaque trailing axis until the final
        # `_reshape_to_logits` splits it.
        if loss_type == "two_hot":
            assert two_hot_bins >= 2, f"two_hot_bins must be >= 2; got {two_hot_bins}"
            assert (
                two_hot_min < two_hot_max
            ), f"two_hot_min ({two_hot_min}) must be < two_hot_max ({two_hot_max})"
            self.num_bins = two_hot_bins
            self.two_hot_min = two_hot_min
            self.two_hot_max = two_hot_max
            self.register_buffer(
                "bin_vals",
                torch.linspace(two_hot_min, two_hot_max, two_hot_bins),
            )
            self._fused_per_step = out_dim * two_hot_bins
        else:
            self.num_bins = 1
            self._fused_per_step = out_dim

        if two_tower:
            assert not self.discrete, "pathways which has no clean discrete analog."
            inp_dim_a = self.state_dim + self.z_dim + 1
            inp_dim_b = self.state_dim + self.action_dim + 1
            self.net = _EinMixTwoTower(
                ensemble_size=self.num_ensemble,
                inp_dim_a=inp_dim_a,
                inp_dim_b=inp_dim_b,
                d_hidden=d_hidden,
                out_dim=self._fused_per_step,
                tower_layers=tower_layers,
                trunk_layers=trunk_layers,
                activation=activation,
                dropout_p=dropout_p,
                input_pre_norm=input_pre_norm,
            )
        else:
            if self.discrete:
                inp_dim = self.state_dim + self.z_dim
                fused_out_dim = self.action_dim * self._fused_per_step * self.num_gammas
            else:
                inp_dim = self.state_dim + self.z_dim + 1 + self.action_dim
                fused_out_dim = self._fused_per_step
            self.net = _EinMixEnsemble(
                ensemble_size=self.num_ensemble,
                inp_dim=inp_dim,
                d_hidden=d_hidden,
                n_layers=n_layers,
                out_dim=fused_out_dim,
                dropout_p=dropout_p,
                activation=activation,
            )

    def __len__(self):
        return self.num_ensemble

    def _fused_forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Heavy lifting shared by ``forward`` and ``forward_logits``.

        Returns shape ``(K, B, L, C, G, _fused_per_step)``. In MSE mode this
        is ``(..., out_dim)`` directly; in two-hot mode it's
        ``(..., out_dim * num_bins)`` and the caller is responsible for
        splitting it via ``_reshape_to_logits``.
        """
        K, B, L, G, _ = action.shape
        assert G == self.num_gammas
        if self.discrete:
            assert K == 1
            inp = torch.cat([state, z], dim=-1)
            out, _ = self.net(inp)
            out = rearrange(
                out,
                "b l c (g d a) -> 1 b l c g d a",
                g=self.num_gammas,
                d=self._fused_per_step,
                a=self.action_dim,
            )
            out = (out * action.unsqueeze(3).unsqueeze(-2)).sum(-1)
        else:
            state_rep = repeat(state, "b l d -> (k b g) l d", k=K, g=self.num_gammas)
            action_rep = rearrange(
                action.clamp(-0.999, 0.999), "k b l g d -> (k b g) l d"
            )
            gammas_rep = gammas_as_input_seq(self.gammas, K * B, L).to(state.device)
            z_rep = repeat(z, "b l d -> (k b g) l d", k=K, g=self.num_gammas)
            if self.two_tower:
                x_a = torch.cat([state_rep, z_rep, gammas_rep], dim=-1)
                x_b = torch.cat([state_rep, action_rep, gammas_rep], dim=-1)
                out, _ = self.net(x_a, x_b)
            else:
                inp = torch.cat([state_rep, z_rep, gammas_rep, action_rep], dim=-1)
                out, _ = self.net(inp)
            out = rearrange(out, "(k b g) l c d -> k b l c g d", k=K, g=self.num_gammas)
        return out

    def _reshape_to_logits(self, fused: torch.Tensor) -> torch.Tensor:
        """Split trailing ``out_dim * num_bins`` into ``(out_dim, num_bins)``."""
        assert self.loss_type == "two_hot"
        return rearrange(
            fused,
            "... (d n) -> ... d n",
            d=self.out_dim,
            n=self.num_bins,
        )

    def _expected_value(self, logits: torch.Tensor) -> torch.Tensor:
        """Per-coord categorical expectation: softmax(logits) @ bin_vals."""
        assert self.loss_type == "two_hot"
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bin_vals).sum(-1)

    def _vector_twohot_labels(self, target: torch.Tensor) -> torch.Tensor:
        """Per-coord two-hot encoding (Dreamer-V3 style).

        target: ``(..., out_dim)``. Assumed already ``.detach()``-ed by the
        caller — no grad path through clamp / one_hot / where.

        Each coord i is encoded independently into a categorical over
        ``self.bin_vals``. Targets outside ``[bin_vals[0], bin_vals[-1]]``
        saturate to the edge bin (all mass on bin 0 or bin n-1); the
        ``TwoHotPhi/Psi/EdgeFraction`` telemetry surfaces this so we can
        widen the range if it happens often.

        Returns: ``(..., out_dim, n_bins)`` probability tensor; sums to 1
        along ``dim=-1``.
        """
        assert self.loss_type == "two_hot"
        bin_vals = self.bin_vals  # (n_bins,)
        target_u = target.unsqueeze(-1)  # (..., out_dim, 1)
        # Per-coord bracketing bins.  Clamp handles out-of-range targets:
        # below=above at the nearest edge bin → equal=True → 0.5/0.5
        # weights collapse onto a single one_hot, summing to 1 on the edge.
        below = ((bin_vals <= target_u).sum(-1) - 1).clamp(0, self.num_bins - 1)
        above = (self.num_bins - (bin_vals > target_u).sum(-1)).clamp(
            0, self.num_bins - 1
        )
        equal = (below == above).unsqueeze(-1)  # (..., out_dim, 1)
        bv = bin_vals.view(-1)
        # When equal: substitute distances of 1.0 each so weights become
        # 0.5/0.5 and the divide-by-zero in `total` is avoided. When not
        # equal: actual distances to the flanking bin centers, summing to
        # bin_width.
        dist_to_below = torch.where(
            equal, 1.0, (bv[below] - target).abs().unsqueeze(-1)
        )
        dist_to_above = torch.where(
            equal, 1.0, (bv[above] - target).abs().unsqueeze(-1)
        )
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total  # (..., out_dim, 1)
        weight_above = dist_to_below / total
        # F.one_hot returns Long; mult with float weights auto-promotes to
        # float.
        return (
            F.one_hot(below, self.num_bins) * weight_below
            + F.one_hot(above, self.num_bins) * weight_above
        )  # (..., out_dim, n_bins)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        log_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        """Returns ``(K, B, L, C, G, out_dim)``. Mode-agnostic.

        - mse:     raw network output.
        - two_hot: expected value of the per-coord softmax(logits) over
                   ``bin_vals``. Fully differentiable through the softmax.
        """
        fused = self._fused_forward(state, action, z)
        if self.loss_type == "mse":
            out = fused
        else:
            logits = self._reshape_to_logits(fused)
            out = self._expected_value(logits)
        if log_dict is not None:
            add_activation_log("FNet-out", out, log_dict)
        return out

    def forward_logits(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        log_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        """Returns ``(K, B, L, C, G, out_dim, num_bins)`` logits. Two-hot only."""
        assert (
            self.loss_type == "two_hot"
        ), "forward_logits requires loss_type='two_hot'"
        fused = self._fused_forward(state, action, z)
        logits = self._reshape_to_logits(fused)
        if log_dict is not None:
            add_activation_log("FNet-logits", logits, log_dict)
        return logits

    def regression_loss(
        self,
        *,
        target: torch.Tensor,
        pred: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Element-wise loss, shape matching ``target`` (``(..., out_dim)``).

        Caller applies its own mask via ``utils.masked_avg``.

        - mse:     ``0.5 * (pred - target).pow(2)``  (requires ``pred``)
        - two_hot: per-coord cross-entropy of ``log_softmax(logits)`` vs the
                   two-hot encoding of ``target``  (requires ``logits``)
        """
        if self.loss_type == "mse":
            assert (
                pred is not None and logits is None
            ), "MSE mode: pass `pred=`, not `logits=`"
            return 0.5 * (pred - target).pow(2)
        else:
            assert (
                logits is not None and pred is None
            ), "two-hot mode: pass `logits=`, not `pred=`"
            target_dist = self._vector_twohot_labels(target)
            log_probs = F.log_softmax(logits, dim=-1)
            return -(target_dist * log_probs).sum(-1)


class JepaEncoder(nn.Module):
    """JEPA target state encoder."""

    def __init__(
        self,
        state_dim: int,
        emb_dim: int = 128,
        d_hidden: int = 256,
        n_layers: int = 2,
        dropout_p: float = 0.0,
        out_norm: str = "fb",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.emb_dim = emb_dim
        self.inp = nn.Linear(state_dim, d_hidden)
        self.inp_norm = Normalization("layer", d_hidden)
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(d_hidden, d_hidden))
            layers.append(nn.LayerNorm(d_hidden))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(d_hidden, emb_dim))
        self.net = nn.Sequential(*layers)
        if out_norm == "fb":
            self.out_norm = FBNormalization(d_model=emb_dim)
        else:
            self.out_norm = Normalization(method=out_norm, d_model=emb_dim)

    @torch.compile
    def forward(
        self,
        state: torch.Tensor,
        log_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        x = torch.tanh(self.inp_norm(self.inp(state)))
        out = self.net(x)
        if log_dict is not None:
            add_activation_log("JepaEncoder-out", out, log_dict)
        out = self.out_norm(out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  AlphaDistribution / AlphaSchedule — controls alpha sampling and scheduling
# ══════════════════════════════════════════════════════════════════════════════


@gin.configurable
class AlphaDistribution(abc.ABC):
    """How to draw alpha (or beta) values in [0, max_value]."""

    @abc.abstractmethod
    def sample(self, shape, device: torch.device) -> torch.Tensor:
        """Returns (*shape, 1) alpha values.  ``shape`` may be an int or a tuple."""

    @property
    def max_value(self) -> float:
        raise NotImplementedError

    @staticmethod
    def _normalize_shape(shape) -> Tuple[int, ...]:
        return (shape,) if isinstance(shape, int) else tuple(shape)


@gin.configurable
class ConstantAlpha(AlphaDistribution):
    def __init__(self, value: float = 0.0):
        self.value = value

    def sample(self, shape, device):
        shape = self._normalize_shape(shape)
        return torch.full((*shape, 1), self.value, device=device)

    @property
    def max_value(self):
        return self.value


@gin.configurable
class ZeroInflatedUniform(AlphaDistribution):
    def __init__(self, max_value: float = 1.0, p_zero: float = 0.5, p_one: float = 0.0):
        assert p_zero + p_one <= 1.0
        self._max_value = max_value
        self.p_zero = p_zero
        self.p_one = p_one

    def sample(self, shape, device):
        shape = self._normalize_shape(shape)
        u = torch.rand(*shape, 1, device=device)
        # [0, p_zero) -> 0,  [p_zero, p_zero+p_one) -> max_value,  else -> Uniform
        alpha = torch.rand(*shape, 1, device=device) * self._max_value
        alpha = torch.where(u < self.p_zero, torch.zeros_like(alpha), alpha)
        alpha = torch.where(
            (u >= self.p_zero) & (u < self.p_zero + self.p_one),
            torch.full_like(alpha, self._max_value),
            alpha,
        )
        return alpha

    @property
    def max_value(self):
        return self._max_value


class AlphaSchedule(abc.ABC):
    """Maps episode_id -> alpha (or beta) values during eval rollouts."""

    @abc.abstractmethod
    def __call__(
        self,
        episode_id: torch.Tensor,
        n_actors: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Returns (n_actors, 1) alpha tensor."""

    @property
    def max_value(self) -> float:
        raise NotImplementedError


@gin.configurable
class ConstantSchedule(AlphaSchedule):
    """Same distribution for all episodes."""

    def __init__(self, dist: AlphaDistribution = None):
        self.dist = dist or ConstantAlpha(0.0)

    def __call__(self, episode_id, n_actors, device):
        return self.dist.sample(n_actors, device)

    @property
    def max_value(self):
        return self.dist.max_value


@gin.configurable
class EpisodeThresholdSchedule(AlphaSchedule):
    """Alpha = high for episode_id < threshold, low afterwards."""

    def __init__(self, high: float = 1.0, low: float = 0.0, threshold: int = 3):
        self._high = high
        self._low = low
        self._threshold = threshold

    def __call__(self, episode_id, n_actors, device):
        high = torch.tensor(self._high, device=device)
        low = torch.tensor(self._low, device=device)
        alpha = torch.where(episode_id < self._threshold, high, low)
        if alpha.ndim == 1:
            alpha = alpha.unsqueeze(-1)
        return alpha

    @property
    def max_value(self):
        return self._high


@gin.configurable
class LinearDecaySchedule(AlphaSchedule):
    """Alpha decays linearly from `high` to `low` over `decay_episodes`, then stays at `low`."""

    def __init__(self, high: float = 1.0, low: float = 0.0, decay_episodes: int = 3):
        self._high = high
        self._low = low
        self._decay_episodes = decay_episodes

    def __call__(self, episode_id, n_actors, device):
        frac = episode_id.float() / max(self._decay_episodes - 1, 1)
        alpha = self._high + (self._low - self._high) * frac.clamp(0.0, 1.0)
        if alpha.ndim == 1:
            alpha = alpha.unsqueeze(-1)
        return alpha

    @property
    def max_value(self):
        return self._high


# β type aliases
BetaDistribution = AlphaDistribution
BetaSchedule = AlphaSchedule


# ══════════════════════════════════════════════════════════════════════════════
#  InferenceMode hierarchy — controls how z is computed / updated at eval time
# ══════════════════════════════════════════════════════════════════════════════


class InferenceMode(abc.ABC):
    """Abstract base for z-inference strategies used by ZeroShotAgent at eval time.

    Subclasses implement how z is initialized, updated per-step, and reset
    between sessions.  The experiment's evaluation loop dispatches to these
    hooks from a per-step rollout loop.
    """

    @abc.abstractmethod
    def init_z(
        self,
        agent: "ZeroShotAgent",
        *,
        n_actors: int,
        device: torch.device,
    ):
        """Compute / sample initial z and set ``agent._inference_z``."""

    def on_step(
        self,
        agent: "ZeroShotAgent",
        phi: torch.Tensor,
        reward: torch.Tensor,
        obs: dict,
    ) -> bool:
        """Called after each env step.  Returns True if z was updated."""
        return False

    def on_session_end(self, agent: "ZeroShotAgent", done_mask: torch.Tensor):
        """Called when ``done=True`` (session boundary).  Reset state for next session."""


@gin.configurable
class OnlineBLR(InferenceMode, abc.ABC):
    """Online BLR-based z inference (abstract base for Thompson sampling).

    Shared hparams mirror the OptiBFM paper defaults.
    """

    def __init__(
        self,
        lam: float = 1.0,
        sigma: float = 0.01,
        decay: float = 1.0,
        kappa: float = 0.0,
        resample: str = "per_step",
    ):
        from metabfm.rl.blr import BLR  # local import to avoid top-level dep

        self.lam = lam
        self.sigma = sigma
        self.decay = decay
        self.kappa = kappa
        assert resample in ("per_step", "per_episode")
        self.resample = resample
        self._blr: Optional["BLR"] = None
        self._prev_episode_id: Optional[torch.Tensor] = None

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _normalize_z(agent: "ZeroShotAgent", z_unnorm: torch.Tensor) -> torch.Tensor:
        return agent.normalize_z(z_unnorm)

    def _set_z(self, agent: "ZeroShotAgent", z_unnorm: torch.Tensor):
        agent._inference_z = self._normalize_z(agent, z_unnorm)

    # -- hooks -------------------------------------------------------------

    def init_z(self, agent, *, n_actors, device, **_kwargs):
        from metabfm.rl.blr import BLR

        self._blr = BLR.create_LSQ(
            dim=agent.task_vector_dim, lam=self.lam, batch_size=n_actors, device=device
        )
        self._prev_episode_id = None
        agent._last_traj_emb_b = None
        z_unnorm = self.select_z(agent, self._blr)
        self._set_z(agent, z_unnorm)

    def on_step(self, agent, phi, reward, obs) -> bool:
        r_scaled = reward * agent.reward_multiplier

        if self.kappa > 0.0:
            phi_scaled = phi / self.sigma if self.sigma != 1.0 else phi
            h = self._blr.cov_norm(phi_scaled).square()
            accept = h >= (math.exp(self.kappa) - 1.0)
            if accept.any():
                self._blr.rank1_update(
                    phi, r_scaled, sigma=self.sigma, decay=self.decay, mask=accept
                )
            # else: skip update entirely
        else:
            self._blr.rank1_update(phi, r_scaled, sigma=self.sigma, decay=self.decay)

        cur_ep = obs.get("episode_id")
        if cur_ep is not None and cur_ep.ndim > 1:
            cur_ep = cur_ep.squeeze(1)

        if self.resample == "per_step":
            do_resample = True
        elif self.resample == "per_episode":
            if self._prev_episode_id is not None and cur_ep is not None:
                do_resample = (cur_ep != self._prev_episode_id).any().item()
            else:
                do_resample = False
        else:
            do_resample = False

        if cur_ep is not None:
            self._prev_episode_id = (
                cur_ep.clone() if isinstance(cur_ep, torch.Tensor) else cur_ep.copy()
            )

        if do_resample:
            z_unnorm = self.select_z(agent, self._blr, phi=phi)
            self._set_z(agent, z_unnorm)
            return True
        return False

    def on_session_end(self, agent, done_mask):
        from metabfm.rl.blr import BLR

        fresh = BLR.create_LSQ(
            dim=self._blr.dim,
            lam=self.lam,
            batch_size=self._blr.batch_size,
            device=self._blr.eta.device,
        )
        for i in range(self._blr.batch_size):
            if done_mask[i]:
                self._blr.eta[i] = fresh.eta[i]
                self._blr.R[i] = fresh.R[i]
        z_unnorm = self.select_z(agent, self._blr)
        self._set_z(agent, z_unnorm)

    @abc.abstractmethod
    def select_z(
        self,
        agent: "ZeroShotAgent",
        blr: "BLR",
        phi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Choose z from the current BLR posterior.  Returns z_unnorm (N, d).

        ``phi`` is the (N, d) feature vector at the current state.  May be
        None on the first call (before any env step has occurred).
        """


@gin.configurable
class ThompsonSampling(OnlineBLR):
    """OptiBFM-TS: z selected via Thompson sampling from BLR posterior."""

    def select_z(self, agent, blr, phi=None):
        return blr.sample()


class RunningMeanStd(nn.Module):
    """EMA-based running variance for intrinsic reward normalization.

    Uses exponential moving average instead of Welford accumulation so that
    the estimate tracks the *recent* scale of the signal rather than the
    all-time average.  ``decay=0.999`` gives an effective window of ~1000
    updates.
    """

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("var", torch.ones(()))
        self.register_buffer("_initialized", torch.tensor(False))

    @torch.no_grad()
    def update(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            valid = x[mask.expand_as(x) > 0]
            if valid.numel() < 2:
                return
            batch_mean = valid.mean()
            batch_var = valid.var(correction=0)
        else:
            batch_mean = x.mean()
            batch_var = x.var(correction=0)
        if not self._initialized:
            self._initialized.fill_(True)
            self.mean.fill_(batch_mean.item())
            self.var.fill_(batch_var.item())
        else:
            self.mean = self.decay * self.mean + (1.0 - self.decay) * batch_mean
            self.var = self.decay * self.var + (1.0 - self.decay) * batch_var

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Mirror amago.PopArtLayer.sigma: clamp before sqrt to floor negative
        # variance from FP drift, then clamp after sqrt to bound the divisor.
        sigma = self.var.clamp(1e-4, 1e8).sqrt().clamp(1e-4, 1e6)
        return x / sigma


def _e3b_scan(phi: torch.Tensor, lam: float) -> torch.Tensor:
    """Cumulative Sherman-Morrison scan for the E3B elliptical bonus.

    Mirrors the ``nstep_return`` pattern in amago.agent: a module-level Python
    loop that ``torch.compile(..., mode="reduce-overhead")`` unrolls into a
    single CUDA graph, binding the loop-controlling scalar (``lam`` here,
    ``n`` there) via ``functools.partial`` at construction time.  ``L`` is
    specialized by Dynamo from the input tensor shape.

    Strict-prefix convention: position 0 has no prior history so ``bonus[:, 0]
    = 0``; position ``t > 0`` uses :math:`C^{-1}_{t-1}` (the Sherman-Morrison
    running inverse after absorbing :math:`\\phi_0..\\phi_{t-1}`).
    """
    B, L, d = phi.shape
    eye = torch.eye(d, device=phi.device, dtype=phi.dtype)
    C_inv = eye.expand(B, d, d).contiguous() / lam  # (B, d, d)
    bonus = phi.new_zeros(B, L, 1)
    for t in range(L):
        phi_t = phi[:, t, :].unsqueeze(-1)  # (B, d, 1)
        Cinv_phi = torch.bmm(C_inv, phi_t)  # (B, d, 1)
        b_t = (phi_t * Cinv_phi).sum(dim=1)  # (B, 1)
        if t > 0:
            bonus[:, t, :] = b_t
        denom = (1.0 + b_t).unsqueeze(-1)  # (B, 1, 1)
        C_inv = C_inv - torch.bmm(Cinv_phi, Cinv_phi.transpose(1, 2)) / denom
    return bonus


def _ngu_scan(
    f: torch.Tensor,
    d_m2: torch.Tensor,
    k: int,
    c: float,
    kernel_eps: float,
    cluster_dist: float,
    max_sim: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized strict-prefix kNN bonus for the NGU episodic reward.

    For each timestep ``t``, takes the ``k`` nearest prior features
    :math:`\\{f_0, ..., f_{t-1}\\}` in squared-Euclidean distance,
    normalizes by the running ``d_m^2``, applies the cluster-distance
    dead-zone and the inverse kernel
    :math:`\\varepsilon / (d_{\\mathrm{norm}} + \\varepsilon)`, sums over
    neighbors, and emits :math:`1/\\sqrt{\\sum + c}`. The bonus is zeroed
    when the kernel sum exceeds ``max_sim`` and at positions ``t < k``
    (too few prior neighbors; the +inf padding would otherwise yield a
    spurious :math:`1/\\sqrt{c}` spike).

    Pure tensor ops (torch-compilable); non-tensor scalars are bound via
    ``functools.partial``.

    Args:
        f: (B, L, d) detached feature tensor (typically IDM).
        d_m2: 0-d running mean of past kNN squared distances. Read-only;
            the caller updates ``ngu_dm2_rms`` from the returned ``d2_topk``.
        k: Number of nearest neighbors.
        c: Pseudo-count regularizer inside the inverse-sqrt.
        kernel_eps: Numerator/floor of the inverse kernel.
        cluster_dist: Subtracted from normalized distance and clamped
            at zero (dead-zone for near-duplicate states).
        max_sim: Bonus zeroed when ``sum kernel > max_sim``.

    Returns:
        bonus: (B, L, 1) detached; zero for ``t < k``.
        d2_topk: (B, L, k) raw squared distances to the k nearest prefix
            neighbors. Cold-start positions carry +inf in unfilled slots;
            the caller masks them (``t >= k``) before updating
            ``ngu_dm2_rms``.
    """
    B, L, _ = f.shape
    # Pairwise squared L2 distances: (B, L, L).
    d2 = torch.cdist(f, f, p=2).pow(2)

    # Strict-prefix mask: invalidate j >= i (exclude self and future).
    arange = torch.arange(L, device=f.device)
    invalid = arange[None, :] >= arange[:, None]  # (L, L), True for j >= i
    INF = torch.finfo(f.dtype).max
    d2 = d2.masked_fill(invalid[None], INF)

    # k smallest distances per (b, t).
    neg_d2_topk, _ = torch.topk(-d2, k, dim=-1)  # (B, L, k)
    d2_topk = -neg_d2_topk

    # Normalize by running d_m^2.  Tiny eps avoids div-by-zero on the
    # first call (before ngu_dm2_rms has been updated); the caller is
    # additionally responsible for zeroing the bonus on that very first
    # call (see compute_intrinsic_reward).
    d_norm = d2_topk / (d_m2 + 1e-8)
    # Cluster-distance dead-zone (collapse near-duplicates to 0).
    d_norm = (d_norm - cluster_dist).clamp(min=0.0)
    # Inverse RBF-style kernel.  Cold-start INF entries -> d_norm huge
    # -> kernel ~0, but cold-start positions are masked out below.
    kernel = kernel_eps / (d_norm + kernel_eps)  # (B, L, k)
    sum_kernel = kernel.sum(dim=-1)  # (B, L)

    bonus = 1.0 / torch.sqrt(sum_kernel + c)

    # Paper's max-similarity outlier cutoff.
    bonus = torch.where(sum_kernel > max_sim, torch.zeros_like(bonus), bonus)

    # Cold-start mask: zero positions with t < k (insufficient neighbors).
    cold = arange < k  # (L,)
    bonus = bonus.masked_fill(cold[None, :], 0.0)

    return bonus.unsqueeze(-1), d2_topk


@gin.configurable
class IntrinsicCriticStack(nn.Module):
    """Critic stack for intrinsic reward Q-learning (Q_P).

    Has no tstep/traj encoder of its own: it consumes the pre-computed
    ``policy_ctx`` tensor produced by the actor's ``PolicyTstepEncoder`` +
    ``traj_encoder_pi`` pass, saving a transformer pass per training step.
    Owns the NCritics ensemble, target critics, PopArt normalization, RND
    networks, and per-bonus running stats.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Dict,
        rl2_space: gym.spaces.Box,
        action_space: gym.spaces.Space,
        max_seq_len: int,
        gammas: torch.Tensor,
        discrete: bool,
        action_dim: int,
        actor: nn.Module,
        policy_ctx_dim: int,
        num_critics: int = 2,
        critic_type: Type[BaseCriticHead] = NCritics,
        tau: float = 0.005,
        n_step: int = 1,
        reward_multiplier: float = 1.0,
        use_popart: bool = True,
        rnd_output_expansion: float = 1.0,
        rnd_hidden_expansion: float = 2.0,
        rnd_n_layers: int = 2,
        rnd_weight_scale: float = 10.0,
        rnd_coeff: float = 0.0,
        rnd_coeff_final: float = 0.0,
        rnd_coeff_decay_steps: Optional[int] = None,
        rldp_error_coeff: float = 1.0,
        rnd_obs_key: Union[str, list[str]] = "rldp",
        e3b_coeff: float = 0.0,
        e3b_lambda: float = 0.1,
        e3b_feature_key: Union[str, list[str]] = "rldp",
        ngu_coeff: float = 0.0,
        ngu_feature_key: Union[str, list[str]] = "idm",
        ngu_k: int = 10,
        ngu_c: float = 0.001,
        ngu_kernel_eps: float = 1e-4,
        ngu_cluster_distance: float = 0.008,
        ngu_max_similarity: float = 8.0,
        use_z_in_critic: bool = False,
        z_dim: int = 0,
        alpha_beta_enc_dim: int = 66,
        ext_ctx_dim: int = 0,
        extrinsic_reward_multiplier: float = 1.0,
        zero_policy_ctx: bool = False,
        n_alphas: int = 1,
        n_betas: int = 1,
        compute_online_q: bool = True,
    ):
        super().__init__()
        self._actor_ref = [actor]
        self._compute_online_q = compute_online_q
        self.gammas = gammas
        self.discrete = discrete
        self.action_dim = action_dim
        self.tau = tau
        self.n_step = n_step
        self.reward_multiplier = reward_multiplier
        self.num_critics = num_critics
        # Used purely for per-corner logging of the raw env-scale TD target /
        # reward in forward(); does NOT affect any compute path.  When > 1 on
        # both axes the leading NAB dim is unfolded into (NA, NB, B, ...) and
        # the (α=1, β=1) corner is logged as a comparable raw-scale metric to
        # the pure-extrinsic baseline's Q.
        self.n_alphas = n_alphas
        self.n_betas = n_betas
        self.rnd_coeff_init = rnd_coeff
        self.rnd_coeff_final = rnd_coeff_final
        self.rnd_coeff_decay_steps = rnd_coeff_decay_steps
        self.use_z_in_critic = use_z_in_critic
        self.z_dim = z_dim
        self.alpha_beta_enc_dim = alpha_beta_enc_dim
        self.ext_ctx_dim = ext_ctx_dim
        self.extrinsic_reward_multiplier = extrinsic_reward_multiplier
        self.zero_policy_ctx = zero_policy_ctx
        if use_z_in_critic and z_dim <= 0:
            raise ValueError(
                "IntrinsicCriticStack.use_z_in_critic=True requires z_dim > 0"
            )
        if (rnd_coeff > 0 or rnd_coeff_final > 0) and rnd_coeff_decay_steps is None:
            raise ValueError(
                "rnd_coeff_decay_steps must be set when rnd_coeff or rnd_coeff_final > 0"
            )
        self.rldp_error_coeff = rldp_error_coeff
        self.rnd_obs_key = rnd_obs_key
        self.e3b_coeff = e3b_coeff
        self.e3b_lambda = e3b_lambda
        self.e3b_feature_key = e3b_feature_key
        self.e3b_feature_dim = (
            self._composite_dim(obs_space, e3b_feature_key) if e3b_coeff > 0 else 0
        )
        self.ngu_coeff = ngu_coeff
        self.ngu_feature_key = ngu_feature_key
        self.ngu_k = ngu_k
        self.ngu_c = ngu_c
        self.ngu_kernel_eps = ngu_kernel_eps
        self.ngu_cluster_distance = ngu_cluster_distance
        self.ngu_max_similarity = ngu_max_similarity
        self.ngu_feature_dim = (
            self._composite_dim(obs_space, ngu_feature_key) if ngu_coeff > 0 else 0
        )
        # Cache max_seq_len so rollout-time NGU memory can be pre-allocated
        # at init_hidden_state without re-querying the trajectory encoder.
        self.max_seq_len = max_seq_len
        self.register_buffer("_rnd_step", torch.zeros((), dtype=torch.long))
        self.update_info: dict = {}
        self._nstep_fn = torch.compile(
            functools.partial(nstep_return, n=n_step), mode="reduce-overhead"
        )

        # State input: the actor's policy_ctx (see TDJepa.compute_losses),
        # Fourier-encoded α + β (alpha_beta_enc_dim dims), and the β-gated
        # extrinsic-context tensor (ext_ctx_dim dims; 0 when β is
        # permanently off and the extrinsic encoder is skipped).
        self.policy_ctx_dim = policy_ctx_dim

        state_dim = (
            policy_ctx_dim
            + alpha_beta_enc_dim
            + ext_ctx_dim
            + (z_dim if use_z_in_critic else 0)
        )
        ac_kwargs = dict(
            state_dim=state_dim,
            action_dim=action_dim,
            discrete=discrete,
            gammas=gammas,
        )
        self.critics = critic_type(**ac_kwargs, num_critics=num_critics)
        self.target_critics = critic_type(**ac_kwargs, num_critics=num_critics)
        self.maximized_critics = critic_type(**ac_kwargs, num_critics=num_critics)
        self._distributional = isinstance(self.critics, NCriticsTwoHot)
        self.popart = PopArtLayer(gammas=len(gammas), enabled=use_popart)
        self.hard_sync_targets()
        self.target_critics.requires_grad_(False)
        self.maximized_critics.requires_grad_(False)

        # RND networks — input dimension determined by rnd_obs_key
        rnd_input_dim = self._composite_dim(obs_space, rnd_obs_key)
        rnd_hidden_dim = round(rnd_hidden_expansion * rnd_input_dim)
        rnd_output_dim = round(rnd_output_expansion * rnd_input_dim)
        self.rnd_prior = self._build_rnd_mlp(
            rnd_input_dim, rnd_hidden_dim, rnd_output_dim, rnd_n_layers
        )
        for p in self.rnd_prior.parameters():
            p.data *= rnd_weight_scale
        self.rnd_prior.requires_grad_(False)
        self.rnd_predictor = self._build_rnd_mlp(
            rnd_input_dim, rnd_hidden_dim, rnd_output_dim, rnd_n_layers
        )

        # Running normalization for intrinsic reward terms
        self.rldp_err_rms = RunningMeanStd()
        self.rnd_rms = RunningMeanStd()
        self.e3b_rms = RunningMeanStd()
        # NGU per-bonus normalizer + running d_m^2 for distance normalization.
        self.ngu_rms = RunningMeanStd()
        self.ngu_dm2_rms = RunningMeanStd()

        # Compiled E3B scan; mirrors self._nstep_fn.  Only built when the
        # bonus is actually used so ordinary runs skip the one-time compile
        # (and avoid allocating any associated CUDA-graph memory).
        if self.e3b_coeff > 0:
            self._e3b_fn = torch.compile(
                functools.partial(_e3b_scan, lam=self.e3b_lambda),
                mode="reduce-overhead",
            )
        else:
            self._e3b_fn = None

        # Compiled NGU scan; same lazy-compile pattern as E3B.  Non-tensor
        # scalars are bound via partial; d_m^2 (a 0-d tensor) is the only
        # runtime kwarg besides f.
        if self.ngu_coeff > 0:
            self._ngu_fn = torch.compile(
                functools.partial(
                    _ngu_scan,
                    k=self.ngu_k,
                    c=self.ngu_c,
                    kernel_eps=self.ngu_kernel_eps,
                    cluster_dist=self.ngu_cluster_distance,
                    max_sim=self.ngu_max_similarity,
                ),
                mode="reduce-overhead",
            )
        else:
            self._ngu_fn = None

    @property
    def actor(self):
        return self._actor_ref[0]

    def _cat_critic_input(
        self,
        policy_ctx: torch.Tensor,
        alpha_enc: torch.Tensor,
        beta_enc: torch.Tensor,
        gated_ext_ctx: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Canonical Q_I critic MLP input.

        Order: ``[policy_ctx, α_enc, β_enc, gated_ext, z?]``.
        Matches ``state_dim = policy_ctx_dim + alpha_beta_enc_dim +
        ext_ctx_dim + z_dim`` from ``__init__``.
        """
        if self.zero_policy_ctx:
            policy_ctx = torch.zeros_like(policy_ctx)
        parts = [policy_ctx, alpha_enc, beta_enc, gated_ext_ctx]
        if self.use_z_in_critic:
            assert z is not None, "use_z_in_critic=True but z is None"
            if self.zero_policy_ctx:
                z = torch.zeros_like(z)
            parts.append(z)
        return torch.cat(parts, dim=-1)

    def normalize_q(self, x: torch.Tensor) -> torch.Tensor:
        """Project a raw env-scale Q into popart-normalized space.

        Same op (``popart.normalize_values``) for both scalar and two-hot
        critics, since IC.forward returns raw env-scale Q uniformly.  Used
        by the online DPG term in ``_policy_actor_loss_step`` to keep
        gradient magnitudes well-conditioned.
        """
        return self.popart.normalize_values(x)

    @property
    def rnd_coeff(self) -> torch.Tensor:
        """Current RND weight as a 0-d device tensor.

        Returning a tensor (rather than ``float(_rnd_step.item())``) avoids a
        blocking GPU->CPU sync on every forward.
        """
        device = self._rnd_step.device
        if self.rnd_coeff_decay_steps is None or self.rnd_coeff_decay_steps <= 0:
            return torch.as_tensor(
                self.rnd_coeff_init, device=device, dtype=torch.float32
            )
        t = self._rnd_step.clamp(max=self.rnd_coeff_decay_steps).to(torch.float32)
        frac = t / self.rnd_coeff_decay_steps
        return self.rnd_coeff_init + (self.rnd_coeff_final - self.rnd_coeff_init) * frac

    @staticmethod
    def _build_rnd_mlp(
        input_dim: int, hidden_dim: int, output_dim: int, n_layers: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        d_in = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.ReLU())
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _gather_feature(
        obs_dict: dict[str, torch.Tensor],
        key_or_keys: Union[str, list[str]],
    ) -> torch.Tensor:
        """Return ``obs_dict[k]`` (single key) or a last-dim concatenation of
        ``obs_dict[k]`` for each k in ``key_or_keys``.

        All gathered tensors must share their leading dims (B, L) and differ
        only along the final feature dim.
        """
        if isinstance(key_or_keys, str):
            return obs_dict[key_or_keys]
        return torch.cat([obs_dict[k] for k in key_or_keys], dim=-1)

    @staticmethod
    def _composite_dim(
        obs_space: gym.spaces.Dict,
        key_or_keys: Union[str, list[str]],
    ) -> int:
        """Sum ``obs_space[k].shape[0]`` over ``key_or_keys`` (str or list)."""
        if isinstance(key_or_keys, str):
            return obs_space[key_or_keys].shape[0]
        return sum(obs_space[k].shape[0] for k in key_or_keys)

    def compute_rnd_bonus(
        self, rnd_input: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RND bonus and predictor loss.

        Args:
            rnd_input: (B, L, D) — keyed by ``self.rnd_obs_key``.
            mask: (B, L, 1) float mask; 0 = padded.

        Returns:
            rnd_bonus: (B, L, 1) detached MSE per-sample
            rnd_loss: scalar predictor loss (masked)
        """
        with torch.no_grad():
            target = self.rnd_prior(rnd_input)
        pred = self.rnd_predictor(rnd_input)
        err = (target - pred).pow(2).mean(dim=-1, keepdim=True)
        if mask is not None:
            loss = utils.masked_avg(err, mask)
        else:
            loss = err.mean()
        return err.detach(), loss

    @torch.no_grad()
    def compute_e3b_bonus(self, phi: torch.Tensor) -> torch.Tensor:
        """Batched training-time E3B scan; thin wrapper around ``self._e3b_fn``.

        Delegates to the compiled ``functools.partial(_e3b_scan, lam=...)``
        registered in ``__init__`` (only built when ``self.e3b_coeff > 0``).

        Args:
            phi: (B, L, d) detached feature tensor.

        Returns:
            bonus: (B, L, 1), position 0 is always zero.
        """
        return self._e3b_fn(phi)

    @torch.no_grad()
    def compute_ngu_bonus(
        self, f: torch.Tensor, d_m2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched training-time NGU strict-prefix kNN scan.

        Thin wrapper around the compiled ``self._ngu_fn`` (only built when
        ``self.ngu_coeff > 0``); see ``_ngu_scan`` for arguments and
        return values. The caller updates ``self.ngu_dm2_rms`` from the
        returned ``d2_topk`` outside the compiled scan.
        """
        return self._ngu_fn(f, d_m2=d_m2)

    @torch.no_grad()
    def compute_e3b_bonus_step(
        self,
        phi_t: torch.Tensor,
        C_inv: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rollout-time single-step E3B update.

        Returns the bonus for this step and the updated inverse covariance.
        When ``valid_mask`` is 0 (first step of a freshly-reset trial), the
        bonus is zeroed to match the training scan's position-0 convention;
        ``C_inv`` is still updated from the reset state, identical to what
        the training scan does at ``t = 0`` (compute b, then update).

        Args:
            phi_t: (B, 1, d) detached feature at current step.
            C_inv: (B, d, d) inverse covariance going into this step; the
                caller (``_BilevelTrajEncoderProxy.reset_hidden_state``) has
                already reset rows corresponding to actors where ``done``.
            valid_mask: (B, 1, 1) float/bool mask; 0 = first step of trial.

        Returns:
            bonus: (B, 1, 1) detached.
            new_C_inv: (B, d, d) after Sherman-Morrison absorption of phi_t.
        """
        phi = phi_t.squeeze(1).unsqueeze(-1)  # (B, d, 1)
        Cinv_phi = torch.bmm(C_inv, phi)  # (B, d, 1)
        b = (phi * Cinv_phi).sum(dim=1, keepdim=True)  # (B, 1, 1)
        denom = 1.0 + b.squeeze(-1).unsqueeze(-1)  # (B, 1, 1)
        new_C_inv = C_inv - torch.bmm(Cinv_phi, Cinv_phi.transpose(1, 2)) / denom
        if valid_mask is not None:
            b = b * valid_mask.to(b.dtype)
        return b, new_C_inv

    @torch.no_grad()
    def compute_ngu_bonus_step(
        self,
        f_t: torch.Tensor,
        ngu_mem: torch.Tensor,
        ngu_step_idx: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rollout-time single-step NGU update with running memory buffer.

        Same computation as ``_ngu_scan``, but neighbors come from a
        per-actor running memory (pre-allocated, zero-initialized) instead
        of within-batch prefixes; ``ngu_step_idx`` counts filled slots.
        Both are reset for done actors in
        ``_BilevelTrajEncoderProxy.reset_hidden_state``.

        Args:
            f_t: (B, 1, d_f) detached feature at current step.
            ngu_mem: (B, max_seq_len, d_f) feature memory.
            ngu_step_idx: (B,) long count of filled slots per actor.
            valid_mask: (B, 1, 1) float/bool mask; 0 = first step of trial.

        Returns:
            bonus: (B, 1, 1) detached.  Zero whenever ``step_idx < ngu_k``
                (cold-start) or when ``valid_mask`` is 0.
            new_mem: (B, max_seq_len, d_f) with ``f_t`` written at slot
                ``min(step_idx, max_seq_len - 1)`` per actor.
            new_step_idx: (B,) incremented by 1, clamped at max_seq_len.
        """
        B = f_t.shape[0]
        M = ngu_mem.shape[1]
        device = ngu_mem.device

        # Distances from f_t to every memory slot.
        f_flat = f_t.squeeze(1)  # (B, d_f)
        dists = (ngu_mem - f_flat.unsqueeze(1)).pow(2).sum(-1)  # (B, M)

        # Mask invalid slots (>= step_idx) with +inf.
        pos = torch.arange(M, device=device).unsqueeze(0)  # (1, M)
        in_range = pos < ngu_step_idx.unsqueeze(1)  # (B, M)
        INF = torch.finfo(ngu_mem.dtype).max
        dists = torch.where(in_range, dists, torch.full_like(dists, INF))

        # k smallest distances.
        neg_topk, _ = torch.topk(-dists, self.ngu_k, dim=-1)  # (B, k)
        d2_topk = -neg_topk

        d_m2 = self.ngu_dm2_rms.mean.detach().to(f_t.dtype)
        d_norm = d2_topk / (d_m2 + 1e-8)
        d_norm = (d_norm - self.ngu_cluster_distance).clamp(min=0.0)
        kernel = self.ngu_kernel_eps / (d_norm + self.ngu_kernel_eps)  # (B, k)
        sum_kernel = kernel.sum(dim=-1)  # (B,)

        bonus = 1.0 / torch.sqrt(sum_kernel + self.ngu_c)  # (B,)
        bonus = torch.where(
            sum_kernel > self.ngu_max_similarity,
            torch.zeros_like(bonus),
            bonus,
        )
        # Cold-start: zero actors whose step_idx < k.
        cold = ngu_step_idx < self.ngu_k  # (B,)
        bonus = torch.where(cold, torch.zeros_like(bonus), bonus)
        # First-call zeroing: d_m^2 not yet initialized.
        if not bool(self.ngu_dm2_rms._initialized.item()):
            bonus = torch.zeros_like(bonus)

        # Append f_t to mem at slot min(step_idx, M - 1) and bump counter.
        new_mem = ngu_mem.clone()
        batch_idx = torch.arange(B, device=device)
        write_idx = ngu_step_idx.clamp(max=M - 1)
        new_mem[batch_idx, write_idx] = f_flat
        new_step_idx = (ngu_step_idx + 1).clamp(max=M)

        bonus_3d = bonus.view(B, 1, 1)
        if valid_mask is not None:
            bonus_3d = bonus_3d * valid_mask.to(bonus_3d.dtype)
        return bonus_3d, new_mem, new_step_idx

    def hard_sync_targets(self):
        _full_copy_inplace(self.target_critics, self.critics)
        _full_copy_inplace(self.maximized_critics, self.critics)

    def soft_sync_targets(self):
        _polyak_ema_inplace(self.target_critics, self.critics, self.tau)
        _full_copy_inplace(self.maximized_critics, self.critics)

    def clip_param_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """Split self.parameters() into ``amago`` and ``rnd`` clip groups.

        ``rnd`` covers only ``rnd_predictor`` (a fully isolated MSE regressor;
        its loss touches no other params and no other loss touches it). Every
        other learnable param in this stack — ``critics``, ``popart``, etc. —
        lands in ``amago`` so it gets clipped together with the actor and the
        policy-side encoders. Frozen members (``target_critics``,
        ``maximized_critics``, ``rnd_prior``) have ``requires_grad=False`` so
        they're naturally excluded by the downstream grad-presence filter.
        """
        rnd_params = list(self.rnd_predictor.parameters())
        rnd_ids = {id(p) for p in rnd_params}
        amago_params = [p for p in self.parameters() if id(p) not in rnd_ids]
        return {"amago": amago_params, "rnd": rnd_params}

    def compute_intrinsic_reward(
        self, batch: Batch, log_step: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-step composite intrinsic reward.

        The policy-visible composite (rldp_error + E3B + NGU) is written to
        ``batch.obs["intrinsic_reward"]`` so ``traj_encoder_pi`` sees it at
        training and rollout time. The TD-target composite adds RND on top,
        so RND trains the intrinsic critic to value globally novel states
        without entering the actor's observation.

        Stateful: updates the per-bonus running stats and increments
        ``self._rnd_step``.  Must be called exactly once per training step,
        BEFORE the actor forward pass.

        Args:
            batch: Full-length Batch (the same length the critic consumes).
                ``batch.obs`` must contain ``self.rnd_obs_key`` and — if
                ``self.rldp_error_coeff > 0`` — ``'rldp_error'``.
            log_step: Whether to populate ``self.update_info``.

        Returns:
            intrinsic_reward_td: ``(B, L, 1)`` detached (with RND).
                Position 0 is zero.
            intrinsic_reward_obs: ``(B, L, 1)`` detached (no RND).
                Position 0 is zero.
            intrinsic_reward_loss: scalar differentiable RND predictor loss.
        """
        self.update_info = {}
        pad_val = MAGIC_PAD_VAL
        full_mask = (~((batch.rl2s == pad_val).all(-1, keepdim=True))).float()
        state_mask = full_mask[:, 1:, ...]

        # ── RND bonus (stateful: updates RMS, increments anneal step) ──
        rnd_input = self._gather_feature(batch.obs, self.rnd_obs_key).detach()
        rnd_bonus, intrinsic_reward_loss = self.compute_rnd_bonus(
            rnd_input, full_mask
        )  # (B, L, 1)
        rnd_shifted = rnd_bonus[:, 1:, :]
        self.rnd_rms.update(rnd_shifted, state_mask)
        cur_rnd_coeff = self.rnd_coeff
        self._rnd_step += 1
        rnd_norm = self.rnd_rms.normalize(rnd_shifted)
        rnd_term = cur_rnd_coeff * rnd_norm

        # obs_shifted accumulates only non-RND terms (visible to the policy).
        # td_shifted adds the RND term on top (critic-only).
        obs_shifted = torch.zeros_like(rnd_term)

        # ── Episode-reset mask (shared by rldp_error + E3B + NGU) ──
        # bonus_shifted[b, i] = bonus for s_{i+1}; ep_change[b, i] = True
        # when s_{i+1} is the first state of a new episode.  Reset
        # transitions are uncontrollable env teleports, so the "novelty"
        # there is a reset artifact, not exploration; zero per-step bonuses
        # there.  Built only when an episodic-style bonus is active.
        ep_change = None
        episodic_reset_mask = None
        if self.e3b_coeff > 0 or self.ngu_coeff > 0 or self.rldp_error_coeff > 0:
            ep_id = batch.obs["episode_id"]
            if ep_id.ndim == 3:
                ep_id = ep_id.squeeze(-1)  # (B, L)
            ep_change = ep_id[:, 1:] != ep_id[:, :-1]  # (B, L-1)
            # Ignore spurious transitions at the valid→padding boundary.
            ep_change = ep_change & (state_mask.squeeze(-1) > 0)
            episodic_reset_mask = (~ep_change).float().unsqueeze(-1)

        err_shifted = None
        rldp_err_norm = None
        if self.rldp_error_coeff > 0:
            rldp_err = batch.obs["rldp_error"].detach()
            err_shifted = rldp_err[:, 1:, :]
            # Mask reset transitions BEFORE the RMS update so the running
            # variance is calibrated on within-episode prediction errors,
            # not the giant teleport-induced spikes at meta-trial resets.
            err_shifted = err_shifted * episodic_reset_mask
            self.rldp_err_rms.update(err_shifted, state_mask)
            rldp_err_norm = self.rldp_err_rms.normalize(err_shifted)
            obs_shifted = obs_shifted + self.rldp_error_coeff * rldp_err_norm

        # ── E3B elliptical episodic bonus ──
        e3b_shifted = None
        e3b_norm = None
        if self.e3b_coeff > 0:
            phi = self._gather_feature(batch.obs, self.e3b_feature_key).detach()
            e3b_bonus = self.compute_e3b_bonus(phi)  # (B, L, 1), pos 0 = 0
            e3b_shifted = e3b_bonus[:, 1:, :]
            e3b_shifted = e3b_shifted * episodic_reset_mask

            self.e3b_rms.update(e3b_shifted, state_mask)
            e3b_norm = self.e3b_rms.normalize(e3b_shifted)
            obs_shifted = obs_shifted + self.e3b_coeff * e3b_norm

        # ── NGU strict-prefix kNN episodic bonus ──
        ngu_shifted = None
        ngu_norm = None
        ngu_dm2_value = None
        if self.ngu_coeff > 0:
            f_ngu = self._gather_feature(batch.obs, self.ngu_feature_key).detach()
            # Use ngu_dm2_rms.mean as the d_m^2 normalizer.  On the very
            # first call it's uninitialized (mean=0), which we handle by
            # zeroing the bonus below; thereafter the running mean tracks
            # the typical kNN squared distance and yields scale-free
            # kernel inputs.
            ngu_dm2_value = self.ngu_dm2_rms.mean.detach().to(f_ngu.dtype)
            ngu_bonus, d2_topk = self.compute_ngu_bonus(
                f_ngu, d_m2=ngu_dm2_value
            )  # (B, L, 1), (B, L, k)
            # First-call zeroing: bonus is meaningless until d_m^2 has
            # been observed at least once.  We still update the d_m^2
            # running stat below so the next batch is well-calibrated.
            if not bool(self.ngu_dm2_rms._initialized.item()):
                ngu_bonus = torch.zeros_like(ngu_bonus)
            ngu_shifted = ngu_bonus[:, 1:, :]
            ngu_shifted = ngu_shifted * episodic_reset_mask

            # Update d_m^2 running stat using only positions with t >= k
            # (cold-start positions carry +inf in their unfilled slots).
            Lm1 = ngu_shifted.shape[1]
            t_idx = torch.arange(Lm1, device=ngu_shifted.device)
            # +1 because shifted indexing means position 0 of shifted
            # corresponds to absolute timestep t=1.
            valid_for_dm2 = (t_idx + 1) >= self.ngu_k  # (L-1,)
            dm2_mask = state_mask * valid_for_dm2.float().view(1, -1, 1)
            d2_topk_shifted = d2_topk[:, 1:, :].mean(dim=-1, keepdim=True)
            self.ngu_dm2_rms.update(d2_topk_shifted, dm2_mask)

            self.ngu_rms.update(ngu_shifted, state_mask)
            ngu_norm = self.ngu_rms.normalize(ngu_shifted)
            obs_shifted = obs_shifted + self.ngu_coeff * ngu_norm

        td_shifted = obs_shifted + rnd_term

        _pad = lambda t: F.pad(t, (0, 0, 1, 0), "constant", 0.0).detach()
        intrinsic_reward_td = _pad(td_shifted)
        intrinsic_reward_obs = _pad(obs_shifted)

        if log_step:
            where_mask = torch.where(state_mask > 0)
            has_valid = where_mask[0].numel() > 0
            _zero = torch.tensor(0.0, device=rnd_norm.device)

            def _masked_min(t: torch.Tensor) -> torch.Tensor:
                return t[where_mask].min() if has_valid else _zero

            def _masked_max(t: torch.Tensor) -> torch.Tensor:
                return t[where_mask].max() if has_valid else _zero

            self.update_info.update(
                {
                    "IntrinsicQ/RND Loss": intrinsic_reward_loss.detach(),
                    "IntrinsicQ/RND Bonus Mean": utils.masked_avg(
                        rnd_shifted, state_mask
                    ),
                    "IntrinsicQ/RND RMS sigma": self.rnd_rms.var.sqrt().item(),
                    "IntrinsicQ/RND Coeff": cur_rnd_coeff,
                    "IntrinsicQ/RND Term (normalized) min": _masked_min(rnd_norm),
                    "IntrinsicQ/RND Term (normalized) max": _masked_max(rnd_norm),
                    "IntrinsicQ/Intrinsic Reward TD Mean": utils.masked_avg(
                        td_shifted, state_mask
                    ),
                    "IntrinsicQ/Intrinsic Reward Obs Mean": utils.masked_avg(
                        obs_shifted, state_mask
                    ),
                    **(
                        {
                            "IntrinsicQ/RLDP Error Mean (raw)": utils.masked_avg(
                                err_shifted, state_mask
                            ),
                            "IntrinsicQ/RLDP Error RMS sigma": self.rldp_err_rms.var.sqrt().item(),
                            "IntrinsicQ/RLDP Error Term (normalized) min": _masked_min(
                                rldp_err_norm
                            ),
                            "IntrinsicQ/RLDP Error Term (normalized) max": _masked_max(
                                rldp_err_norm
                            ),
                        }
                        if self.rldp_error_coeff > 0
                        else {}
                    ),
                    **(
                        {
                            "IntrinsicQ/E3B Bonus Mean (raw)": utils.masked_avg(
                                e3b_shifted, state_mask
                            ),
                            "IntrinsicQ/E3B RMS sigma": self.e3b_rms.var.sqrt().item(),
                            "IntrinsicQ/E3B Coeff": self.e3b_coeff,
                            "IntrinsicQ/E3B Term (normalized) min": _masked_min(
                                e3b_norm
                            ),
                            "IntrinsicQ/E3B Term (normalized) max": _masked_max(
                                e3b_norm
                            ),
                        }
                        if self.e3b_coeff > 0
                        else {}
                    ),
                    **(
                        {
                            "IntrinsicQ/NGU Bonus Mean (raw)": utils.masked_avg(
                                ngu_shifted, state_mask
                            ),
                            "IntrinsicQ/NGU RMS sigma": self.ngu_rms.var.sqrt().item(),
                            "IntrinsicQ/NGU Coeff": self.ngu_coeff,
                            "IntrinsicQ/NGU d_m^2": self.ngu_dm2_rms.mean.item(),
                            "IntrinsicQ/NGU Term (normalized) min": _masked_min(
                                ngu_norm
                            ),
                            "IntrinsicQ/NGU Term (normalized) max": _masked_max(
                                ngu_norm
                            ),
                        }
                        if self.ngu_coeff > 0
                        else {}
                    ),
                    **(
                        {
                            "IntrinsicQ/Episodic Reset-Zeroed Steps": ep_change.sum().item(),
                        }
                        if (self.e3b_coeff > 0 or self.ngu_coeff > 0)
                        else {}
                    ),
                }
            )

        return intrinsic_reward_td, intrinsic_reward_obs, intrinsic_reward_loss

    @torch.no_grad()
    def compute_intrinsic_reward_step(
        self,
        obs: dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
        e3b_sigma_inv: Optional[torch.Tensor] = None,
        episodic_reset_mask: Optional[torch.Tensor] = None,
        ngu_mem: Optional[torch.Tensor] = None,
        ngu_step_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Rollout-time variant of compute_intrinsic_reward (single timestep).

        Returns the obs-visible intrinsic reward (rldp_error + E3B + NGU,
        no RND), matching training's ``intrinsic_reward_obs``.

        Differences from the training version: runs at (B, 1, *); uses
        frozen running stats (they only advance during training); skips the
        RND predictor loss; and zeros the composite via ``valid_mask`` at
        the first step of a freshly-reset trial (training's position-0 pad).

        Args:
            obs: Single-step obs dict.
            valid_mask: Optional (B, 1, 1) bool/float mask; 0 at the first step
                of a freshly-reset trial.
            episodic_reset_mask: Optional (B, 1, 1) float mask; 0 at
                episode-reset boundaries to zero the rldp_error, E3B, and NGU
                components, as in training.
            ngu_mem: Optional (B, max_seq_len, d_f) NGU memory buffer.
                Required when ``self.ngu_coeff > 0``.
            ngu_step_idx: Optional (B,) long counter of filled slots.
                Required when ``self.ngu_coeff > 0``.

        Returns:
            intrinsic_reward: (B, 1, 1) detached float tensor (no RND).
            new_e3b_sigma_inv: updated (B, d, d) after Sherman-Morrison
                absorption, or ``e3b_sigma_inv`` unchanged when E3B is off.
            new_ngu_mem: updated NGU memory after appending current feature,
                or ``ngu_mem`` unchanged when NGU is off.
            new_ngu_step_idx: incremented step counter, or ``ngu_step_idx``
                unchanged when NGU is off.
        """
        rldp_ref = obs["rldp"]
        intrinsic = torch.zeros(rldp_ref.shape[0], 1, 1, device=rldp_ref.device)

        if self.rldp_error_coeff > 0:
            rldp_err_term = self.rldp_error_coeff * self.rldp_err_rms.normalize(
                obs["rldp_error"]
            )
            # Match training: zero rldp_error at episode-reset boundaries.
            if episodic_reset_mask is not None:
                rldp_err_term = rldp_err_term * episodic_reset_mask
            intrinsic = intrinsic + rldp_err_term

        new_sigma_inv = e3b_sigma_inv
        if self.e3b_coeff > 0:
            phi_t = self._gather_feature(
                obs, self.e3b_feature_key
            ).detach()  # (B, 1, d)
            e3b_bonus, new_sigma_inv = self.compute_e3b_bonus_step(
                phi_t, e3b_sigma_inv, valid_mask=valid_mask
            )
            e3b_term = self.e3b_coeff * self.e3b_rms.normalize(e3b_bonus)
            if episodic_reset_mask is not None:
                e3b_term = e3b_term * episodic_reset_mask
            intrinsic = intrinsic + e3b_term

        new_ngu_mem = ngu_mem
        new_ngu_step_idx = ngu_step_idx
        if self.ngu_coeff > 0:
            assert ngu_mem is not None and ngu_step_idx is not None, (
                "compute_intrinsic_reward_step requires ngu_mem / ngu_step_idx "
                "when ngu_coeff > 0; caller must allocate them via "
                "_BilevelTrajEncoderProxy.init_hidden_state."
            )
            f_t = self._gather_feature(obs, self.ngu_feature_key).detach()
            ngu_bonus, new_ngu_mem, new_ngu_step_idx = self.compute_ngu_bonus_step(
                f_t, ngu_mem, ngu_step_idx, valid_mask=valid_mask
            )
            ngu_term = self.ngu_coeff * self.ngu_rms.normalize(ngu_bonus)
            if episodic_reset_mask is not None:
                ngu_term = ngu_term * episodic_reset_mask
            intrinsic = intrinsic + ngu_term

        if valid_mask is not None:
            intrinsic = intrinsic * valid_mask.to(dtype=intrinsic.dtype)
        return intrinsic.detach(), new_sigma_inv, new_ngu_mem, new_ngu_step_idx

    def forward(
        self,
        policy_ctx_ab: torch.Tensor,
        batch: Batch,
        intrinsic_reward: torch.Tensor,
        a_agent: torch.Tensor,
        a_prime: torch.Tensor,
        alpha_enc: torch.Tensor,
        beta_enc: torch.Tensor,
        beta_raw: torch.Tensor,
        gated_ext_ctx: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        log_step: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the intrinsic critic TD pass (mirrors AMAGO Agent.forward).

        Consumes a pre-computed ``intrinsic_reward``; callers MUST invoke
        ``compute_intrinsic_reward`` first (once per training step).

        Encoder sharing: the IC has no tstep/traj encoder of its own.
        ``policy_ctx_ab`` comes from ``TDJepa.tstep_encoder_pi`` +
        ``TDJepa.traj_encoder_pi``; only (α, β)-conditioned head
        computations happen here.

        Args:
            policy_ctx_ab: Shared policy context, ``(NAB*B, L, policy_ctx_dim)``,
                already repeated NA*NB times by the caller. Gradients from
                the IC critic loss flow back into the actor's encoders.
            batch: Non-replicated Batch of shape ``(B, L, ...)``; supplies
                ``actions``, ``dones``, and ``rl2s`` for masks / buffered
                actions / TD targets.
            intrinsic_reward: Non-replicated ``(B, L, 1)`` detached tensor
                from ``compute_intrinsic_reward``; the transition reward
                used here is ``intrinsic_reward[:, 1:]``.
            a_agent: Actions from TDJepa's online actor,
                (K, NAB*B, L, G, D).
            a_prime: Actions from TDJepa's target actor,
                (K_c, NAB*B, L-1, G, D).
            alpha_enc: Fourier-encoded per-element α values,
                (NAB*B, L, alpha_beta_enc_dim/2), concatenated to the critic
                state input.  NAB inferred as ``alpha_enc.shape[0] // B``.
            beta_enc: Fourier-encoded per-element β values,
                (NAB*B, L, alpha_beta_enc_dim/2), concatenated to the critic
                state input.
            beta_raw: Raw per-element β values, (NAB*B, L, 1).  Used ONLY
                for the TD-target reward mix (positions 1..); the critic MLP
                never sees the raw scalar.
            gated_ext_ctx: ``β · ext_ctx`` (already gated by the caller).
                (NAB*B, L, ext_ctx_dim).  Empty (last dim 0) when the
                extrinsic encoder is auto-skipped.
            z: Optional per-element task latent, (NAB*B, L, z_dim).
                Required iff self.use_z_in_critic.
            log_step: Whether to populate self.update_info with stats.

        Returns:
            (critic_loss, q_s_a_agent, q_s_a) at NAB*B scale.
        """
        pad_val = MAGIC_PAD_VAL

        B, L, _ = batch.rl2s.shape
        NAB = alpha_enc.shape[0] // B
        assert (
            alpha_enc.shape[0] == NAB * B
        ), f"alpha_enc.shape[0]={alpha_enc.shape[0]} not divisible by B={B}"
        assert (
            beta_enc.shape[0] == NAB * B
        ), f"beta_enc.shape[0]={beta_enc.shape[0]} != NAB*B={NAB * B}"
        assert beta_raw.shape == (
            NAB * B,
            L,
            1,
        ), f"beta_raw shape {tuple(beta_raw.shape)} != (NAB*B={NAB * B}, L={L}, 1)"
        assert policy_ctx_ab.shape[:2] == (NAB * B, L), (
            f"policy_ctx_ab shape {tuple(policy_ctx_ab.shape)} does not match "
            f"expected (NAB*B={NAB * B}, L={L}, ...)"
        )
        assert gated_ext_ctx.shape[:2] == (NAB * B, L), (
            f"gated_ext_ctx shape {tuple(gated_ext_ctx.shape)} does not match "
            f"expected (NAB*B={NAB * B}, L={L}, ...)"
        )

        def _rep_ab(v: torch.Tensor) -> torch.Tensor:
            return repeat(v, "b ... -> (a b) ...", a=NAB) if NAB > 1 else v

        # ── Masks (built at (B, L); rep_ab'd below) ──
        full_mask = (~((batch.rl2s == pad_val).all(-1, keepdim=True))).float()
        state_mask = full_mask[:, 1:, ...]

        # ── Transition-aligned intrinsic reward (position t = reward for s_{t+1}) ──
        intrinsic_rews = intrinsic_reward[:, 1:, :]

        G = len(self.gammas)
        C = self.num_critics

        # ── Buffer actions + reward/done tensors, built at (B, L) then rep_ab'd ──
        a = batch.actions
        a = a.clamp(0, 1.0) if self.discrete else a.clamp(-1.0, 1.0)
        a_buffer = F.pad(a, (0, 0, 0, 1), "replicate")
        a_buffer = repeat(a_buffer, f"b l a -> b l {G} a")
        a_buffer = self.actor.policy_dist.action_from_buffer(a_buffer)

        # Intrinsic reward channel (kept at the IC's reward_multiplier scale).
        r_int = repeat(
            (self.reward_multiplier * intrinsic_rews).float(),
            f"b l r -> b l 1 {G} r",
        )
        # Extrinsic reward channel (already at env scale, scaled by
        # extrinsic_reward_multiplier — separate from reward_multiplier so
        # the two channels can be balanced independently).
        r_ext_raw = (batch.rews * self.extrinsic_reward_multiplier).float()
        r_ext = repeat(r_ext_raw, f"b l r -> b l 1 {G} r")
        d = repeat(batch.dones.float(), f"b l d -> b l 1 {G} d")
        gamma = self.gammas.to(r_int.device).unsqueeze(-1)

        # Replicate alpha-independent tensors to (NAB*B, L, ...) for the critic heads.
        a_buffer_ab = _rep_ab(a_buffer)  # (NAB*B, L, G, A_dim)
        r_int_ab = _rep_ab(r_int)  # (NAB*B, L-1, 1, G, 1)
        r_ext_ab = _rep_ab(r_ext)  # (NAB*B, L-1, 1, G, 1)
        d_ab = _rep_ab(d)  # (NAB*B, L, 1, G, 1)
        state_mask_ab = _rep_ab(state_mask)  # (NAB*B, L-1, 1)
        critic_mask = repeat(
            state_mask_ab, f"b l 1 -> b l {C} {G} 1"
        )  # (NAB*B, L-1, C, G, 1)

        # ── (1 - β) · r_int + β · r_ext mix on the TD-target side ──
        # beta_raw is (NAB*B, L, 1); slice to L-1 and broadcast across (1, G, 1)
        # to match r_int_ab / r_ext_ab.  At β=0 cells: r_mix = r_int (extrinsic
        # reward is architecturally invisible).  At β=1 cells: r_mix = r_ext
        # (intrinsic reward is gone).
        beta_raw_td = beta_raw[:, 1:, :]  # (NAB*B, L-1, 1)
        beta_raw_td = beta_raw_td.unsqueeze(2).unsqueeze(-1)  # (NAB*B, L-1, 1, 1, 1)
        r_ab = (1.0 - beta_raw_td) * r_int_ab + beta_raw_td * r_ext_ab

        # ── Concatenate Fourier-encoded α + β + gated ext_ctx (and optional z) ──
        s_rep_a = self._cat_critic_input(
            policy_ctx_ab,
            alpha_enc,
            beta_enc,
            gated_ext_ctx,
            z,
        )

        # ── Q(s, a_buffer) — self.critics (AMAGO's Agent.forward /
        # MultiTaskAgent.forward).  Returned tensor `q_s_a` is in
        # raw env-scale uniformly: scalar -> popart-denormalized, two-hot ->
        # bin_dist_to_raw_vals.
        if self._distributional:
            # Two-hot: feed full L (as in AMAGO's MultiTaskAgent.forward),
            # then post-slice to L-1 after bin_dist_to_raw_vals.
            q_dist_buffer = self.critics(s_rep_a, a_buffer_ab.unsqueeze(0))
            # bin_dist_to_raw_vals returns (K=1, NAB*B, L, C, G, 1)
            q_s_a = self.critics.bin_dist_to_raw_vals(q_dist_buffer).squeeze(0)[
                :, :-1, ...
            ]
        else:
            # Scalar: pre-slice input, then popart-denormalize the linear
            # output to get raw env-scale.
            q_dist_buffer = self.critics(
                s_rep_a[:, :-1, ...], a_buffer_ab[:, :-1, ...].unsqueeze(0)
            )
            q_s_a = self.popart(q_dist_buffer.mean(0), normalized=False)

        # ── Q(s, a_agent) — follows AMAGO's Agent.forward recycling
        # pattern.  When self._compute_online_q (i.e., online_coeff > 0), use
        # the hard-synced `maximized_critics` so the DPG term has gradients
        # while the offline V(s) (computed from the same tensor with .detach())
        # remains correct.  When online_coeff == 0 the maximized_* copy is
        # dead code; we use `self.critics` in no_grad to skip the autograd
        # graph + per-step _full_copy bookkeeping.  In both cases the output
        # `q_s_a_agent` is in raw env-scale.
        if self._compute_online_q:
            q_dist_agent = self.maximized_critics(s_rep_a.detach(), a_agent)
            if self._distributional:
                q_s_a_agent = self.maximized_critics.bin_dist_to_raw_vals(
                    q_dist_agent
                ).mean(0)
            else:
                q_s_a_agent = self.popart(q_dist_agent.mean(0), normalized=False)
        else:
            with torch.no_grad():
                q_dist_agent = self.critics(s_rep_a.detach(), a_agent)
                if self._distributional:
                    q_s_a_agent = self.critics.bin_dist_to_raw_vals(q_dist_agent).mean(
                        0
                    )
                else:
                    q_s_a_agent = self.popart(q_dist_agent.mean(0), normalized=False)

        # ── TD targets with a_prime (AMAGO's Agent.forward /
        # MultiTaskAgent.forward).  Target critic output is reduced
        # to raw env-scale (scalar -> popart denorm; two-hot -> bin_dist_to_raw_vals)
        # and fed to the n-step backup; popart.update_stats(td_target) is kept
        # in both modes to maintain `mu`/`sigma` for downstream actor
        # normalization (see `IntrinsicCriticStack.normalize_q`).
        # a_prime is (K_c, NAB*B, L-1, G, D) where a_prime[t] = π_target(s_{t+1}),
        # matching s_rep[:, 1:] directly (no further slicing needed).
        with torch.no_grad():
            target_dist = self.target_critics(s_rep_a[:, 1:, ...], a_prime)
            if self._distributional:
                q_targ = self.target_critics.bin_dist_to_raw_vals(target_dist).mean(0)
            else:
                q_targ = self.popart(target_dist.mean(0), normalized=False)
            q_reduced = self._reduce_ensemble(q_targ)
            nstep_mask = state_mask_ab.float().unsqueeze(-1).unsqueeze(-1)
            td_target = self._nstep_fn(r_ab, d_ab, q_reduced, gamma, mask=nstep_mask)
            self.popart.update_stats(td_target, mask=critic_mask.all(2, keepdim=True))
            if self._distributional:
                # Same as AMAGO's MultiTaskAgent.forward.
                td_target_labels = self.target_critics.raw_vals_to_labels(td_target)
                td_target_labels = repeat(
                    td_target_labels, f"b l 1 g bins -> b l {C} g bins"
                )
            else:
                td_target_norm = self.popart.normalize_values(td_target)

        # ── Critic loss ──
        if not self._distributional:
            # Scalar path — MSE in popart-normalized space.
            critic_loss = (
                self.popart(q_dist_buffer.mean(0)) - td_target_norm.detach()
            ).pow(2)
        else:
            # Two-hot path — same as AMAGO's MultiTaskAgent.forward.
            B_eff = q_dist_buffer.logits.shape[1]
            G_eff = q_dist_buffer.logits.shape[4]
            Bins = q_dist_buffer.logits.shape[-1]
            ce = F.cross_entropy(
                rearrange(
                    q_dist_buffer.logits[0, :, :-1, ...], "b l c g u -> (b l c g) u"
                ),
                rearrange(td_target_labels, "b l c g u -> (b l c g) u"),
                reduction="none",
            )
            critic_loss = rearrange(
                ce, "(b l c g) -> b l c g 1", b=B_eff, l=L - 1, c=C, g=G_eff
            )
        critic_loss_scalar = utils.masked_avg(critic_loss, critic_mask)

        if log_step:
            r_mask = critic_mask.all(2, keepdim=True)
            where_mask = torch.where(r_mask > 0)
            has_valid = where_mask[0].numel() > 0
            self.update_info.update(
                {
                    "IntrinsicQ/Q(s,a) mean": utils.masked_avg(q_s_a, critic_mask),
                    "IntrinsicQ/Q(s,a) ensemble std": q_s_a.std(2).mean(),
                    "IntrinsicQ/TD Target min": (
                        td_target[where_mask].min() if has_valid else torch.tensor(0.0)
                    ),
                    "IntrinsicQ/TD Target max": (
                        td_target[where_mask].max() if has_valid else torch.tensor(0.0)
                    ),
                    "IntrinsicQ/TD Target mean": utils.masked_avg(td_target, r_mask),
                    "IntrinsicQ/Mean Reward": utils.masked_avg(r_ab, r_mask)
                    / self.reward_multiplier,
                }
            )

            # ── Per-corner raw-scale TD target / reward (apples-to-apples
            # vs the pure-extrinsic baseline's IntrinsicQ/TD Target mean) ──
            # td_target / r_ab / r_mask are all at the IC's raw env scale
            # before PopArt, so the (α=1, β=1) corner here is the directly
            # comparable extrinsic-Q target series.  Convention matches
            # _policy_actor_loss_step.corner_slices: slot 0 = α=0/β=0,
            # slot -1 = α=max/β=max.  For pure_extrinsic (NA=NB=1) these
            # collapse to a single "alpha0" series equal to the aggregate.
            NA, NB = self.n_alphas, self.n_betas
            corner_slices = [("alpha0", 0, 0)]
            if NA >= 2:
                corner_slices.append(("alpha_max", -1, 0))
            if NB >= 2:
                # β=max corner at the pinned-α slot (slot 0).  Useful for
                # intrinsic_meta_rl (NA=1, NB≥2) where slot 0 of α is the
                # α=1 pin and slot -1 of β is the β=1 (AMAGO-equivalent)
                # cell.  For metabfm_alphabeta (NA≥2) this complements
                # alpha_max_beta_max with the β-only pure-extrinsic corner.
                corner_slices.append(("beta_max", 0, -1))
            if NA >= 2 and NB >= 2:
                corner_slices.append(("alpha_max_beta_max", -1, -1))
            td_target_g = rearrange(
                td_target,
                "(a c b) l n g 1 -> a c b l n g 1",
                a=NA,
                c=NB,
            )
            r_ab_g = rearrange(
                r_ab,
                "(a c b) l n g 1 -> a c b l n g 1",
                a=NA,
                c=NB,
            )
            r_mask_g = rearrange(
                r_mask,
                "(a c b) l n g 1 -> a c b l n g 1",
                a=NA,
                c=NB,
            )
            # Restrict to the deployment gamma head (g=-1, γ=TDJepa.gamma).
            # Averaging over the multigamma G axis would mix very different
            # bootstrap horizons.  g=-1 is directly comparable to
            # ActorLoss/Q_P_{name} and to the pure_extrinsic baseline's
            # aggregate (G=1 there).
            for name, ai, ci in corner_slices:
                td_dep = td_target_g[ai, ci, ..., -1:, :]
                r_dep = r_ab_g[ai, ci, ..., -1:, :]
                m_dep = r_mask_g[ai, ci, ..., -1:, :]
                self.update_info[f"IntrinsicQ/TD Target {name}"] = utils.masked_avg(
                    td_dep,
                    m_dep,
                )
                self.update_info[f"IntrinsicQ/Mean Reward {name}"] = (
                    utils.masked_avg(r_dep, m_dep) / self.reward_multiplier
                )

            self.update_info.update(
                {
                    "IntrinsicQ/Min Reward": (
                        r_ab[where_mask].min() / self.reward_multiplier
                        if has_valid
                        else torch.tensor(0.0)
                    ),
                    "IntrinsicQ/Max Reward": (
                        r_ab[where_mask].max() / self.reward_multiplier
                        if has_valid
                        else torch.tensor(0.0)
                    ),
                    "IntrinsicQ/Sequences Containing Done": (d_ab * r_mask)
                    .any((1, 2, 3))
                    .sum(),
                    "IntrinsicQ/Critic Loss": critic_loss_scalar.detach(),
                    "IntrinsicQ/PopArt mu": self.popart.mu.data.mean().item(),
                    "IntrinsicQ/PopArt sigma": self.popart.sigma.mean().item(),
                    # Reward-component stats (RND Loss, RND Bonus Mean, RND RMS sigma,
                    # RND Coeff, RLDP Error Mean/RMS, Intrinsic Reward Mean) are
                    # populated by compute_intrinsic_reward() which runs before
                    # forward() and initializes self.update_info for this step.
                }
            )
            # Per-gamma diagnostics (mirrors AMAGO ``Agent._td_stats``). With
            # multigamma actor + Q_P, individual heads can train at very
            # different value scales — log them per gamma so we can see
            # whether the multigamma trick is amplifying gradients
            # uniformly or whether one head is dominating PopArt.  PopArt
            # ``mu``/``sigma`` are per-gamma buffers of shape ``(G, 1)``.
            popart_sigma_per_g = self.popart.sigma.detach().squeeze(-1)
            popart_mu_per_g = self.popart.mu.data.detach().squeeze(-1)
            for i, g in enumerate(self.gammas):
                gtag = f"{float(g):.3f}"
                self.update_info[f"IntrinsicQ/Q(s,a) gamma={gtag}"] = utils.masked_avg(
                    q_s_a[..., i : i + 1, :],
                    critic_mask[..., i : i + 1, :],
                )
                self.update_info[f"IntrinsicQ/PopArt sigma gamma={gtag}"] = (
                    popart_sigma_per_g[i].item()
                )
                self.update_info[f"IntrinsicQ/PopArt mu gamma={gtag}"] = (
                    popart_mu_per_g[i].item()
                )

            # Static config diagnostics — surfaces critic_type, recycling-mode,
            # and (when distributional) two-hot bin range / count to wandb so
            # the dashboard always reflects which IC critic flavor is active.
            self.update_info["IntrinsicQ/critic_type"] = type(self.critics).__name__
            self.update_info["IntrinsicQ/compute_online_q"] = float(
                self._compute_online_q
            )
            if self._distributional:
                bin_vals = self.critics.bin_vals
                self.update_info["IntrinsicQ/two_hot_min_return"] = float(
                    bin_vals[0].item()
                )
                self.update_info["IntrinsicQ/two_hot_max_return"] = float(
                    bin_vals[-1].item()
                )
                self.update_info["IntrinsicQ/two_hot_output_bins"] = float(
                    self.critics.num_bins
                )

                # Argmax-bin histograms (as in AMAGO's
                # MultiTaskAgent._td_stats).  Tells us whether the bin range
                # [min_return, max_return] is right-sized: pileup at bin 0
                # / Bins-1 means the support is too narrow; a tight middle
                # cluster means the bin support is unnecessarily wide.
                # `q_dist_buffer.logits` is (K=1, NAB*B, L, C, G, Bins);
                # slice K=0 + L → L-1 to match `critic_mask`.
                try:
                    import wandb as _wandb

                    raw_q_bins = q_dist_buffer.logits[
                        0, :, :-1, ...
                    ]  # (NAB*B, L-1, C, G, Bins)
                    Bins = raw_q_bins.shape[-1]
                    cm = critic_mask  # (NAB*B, L-1, C, G, 1)
                    max_all = raw_q_bins.argmax(-1)[torch.where(cm.all(-1))]
                    max_targ = raw_q_bins[..., -1, :].argmax(-1)[
                        torch.where(cm[..., -1, :].all(-1))
                    ]
                    if max_all.numel() > 0:
                        self.update_info["IntrinsicQ/Maximum Bin (All Gammas)"] = (
                            _wandb.Histogram(
                                max_all.detach().cpu().numpy(), num_bins=Bins
                            )
                        )
                    if max_targ.numel() > 0:
                        self.update_info["IntrinsicQ/Maximum Bin (Target Gamma)"] = (
                            _wandb.Histogram(
                                max_targ.detach().cpu().numpy(), num_bins=Bins
                            )
                        )
                except Exception:
                    # Histogram is purely diagnostic; never let a wandb
                    # import / serialization issue kill the train step.
                    pass

        return critic_loss_scalar, q_s_a_agent, q_s_a

    def _reduce_ensemble(self, q: torch.Tensor) -> torch.Tensor:
        """Min over a random 2-critic subset (clipped double-Q)."""
        B, L, C, G, _ = q.shape
        idx = torch.randint(0, C, (B, L, min(2, C), G, 1), device=q.device)
        subset = torch.take_along_dim(q, idx, dim=2)
        return subset.min(2, keepdim=True).values


class ZeroShotAgent(BaseAgent, abc.ABC):
    """Base class for zero-shot RL agents (TDJepa).

    Dynamics/LDP is handled externally by DynamicsLearner via BilevelAgent.
    This class only handles the RL loss (critic + actor).
    """

    def __init__(
        self,
        obs_space: gym.spaces.Dict,
        rl2_space: gym.spaces.Box,
        action_space: gym.spaces.Space,
        max_seq_len: int,
        tstep_encoder_type: Type[TstepEncoder] = gin.REQUIRED,
        traj_encoder_type: Type[TrajEncoder] = gin.REQUIRED,
        # Second backbone (network B / psi)
        tstep_encoder_b_type: Type[TstepEncoder] = gin.REQUIRED,
        traj_encoder_b_type: Type[TrajEncoder] = gin.REQUIRED,
        # Per-backbone observation key selection (passed directly to tstep encoders)
        specify_obs_keys: list[str] = gin.REQUIRED,
        specify_obs_keys_b: list[str] = gin.REQUIRED,
        # Dual sequence length: MLP losses use a subsampled window of this length
        mlp_seq_len: int = gin.REQUIRED,
        # Required intermediate sequence length for actor / intrinsic critic
        # computations.  Slices batch down from max_seq_len to ac_seq_len in
        # BilevelAgent.forward after the dynamics learner + intrinsic reward
        # run at the full length.  Also sizes the K/V cache of the AC traj
        # encoders (phi/psi/pi) at rollout, so the policy never attends to
        # more context than it saw at training time.  Required relation:
        #   mlp_seq_len <= ac_seq_len <= max_seq_len.
        ac_seq_len: int = gin.REQUIRED,
        # RL config
        num_critics: int = 2,
        num_critics_td: int = 2,
        reward_multiplier: float = 10.0,
        online_coeff: float = 1.0,
        offline_coeff: float = 0.0,
        sf_loss_weight: float = 10.0,
        gamma: float = 0.98,
        sf_gamma: float = 0.97,
        tau: float = 0.002,
        fake_filter: bool = False,
        num_actions_for_value_in_critic_loss: int = 1,
        num_actions_for_value_in_actor_loss: int = 1,
        fbc_filter_func: callable = binary_filter,
        use_target_actor: bool = True,
        use_multigamma: bool = False,
        pass_obs_keys_to_actor: Optional[Iterable[str]] = None,
        fb_ortho_coeff: float = 1.0,
        fb_max_pairs: int = 0,
        ortho_mode: str = "all",
        alpha_dist: AlphaDistribution = None,
        val_alpha_schedule: AlphaSchedule = None,
        beta_dist: BetaDistribution = None,
        val_beta_schedule: BetaSchedule = None,
        intrinsic_critic_loss_weight: float = 1.0,
        pin_pure_extrinsic_fraction: float = 0.0,
    ):
        # Store obs keys before super().__init__() which calls init_encoders().
        self._specify_obs_keys = specify_obs_keys
        self._specify_obs_keys_b = specify_obs_keys_b
        self._tstep_encoder_b_type = tstep_encoder_b_type
        self._traj_encoder_b_type = traj_encoder_b_type

        # Validate the seq-len ladder up front and stash before
        # super().__init__() so init_encoders() can size the AC traj encoder
        # K/V caches to ac_seq_len rather than the full max_seq_len.
        assert mlp_seq_len <= ac_seq_len <= max_seq_len, (
            f"ac_seq_len ({ac_seq_len}) must satisfy "
            f"mlp_seq_len ({mlp_seq_len}) <= ac_seq_len <= max_seq_len ({max_seq_len})"
        )
        self.mlp_seq_len = mlp_seq_len
        self.ac_seq_len = ac_seq_len

        super().__init__(
            obs_space=obs_space,
            rl2_space=rl2_space,
            action_space=action_space,
            max_seq_len=max_seq_len,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
        )
        self.tstep_encoder.hide_rl2s = True

        multigammas = (
            Multigammas().discrete if self.discrete else Multigammas().continuous
        )
        gammas = (multigammas if use_multigamma else []) + [gamma]
        self.gammas = torch.Tensor(gammas).float()
        # Single-element gamma tensor for the SF / TDJepa backup. Lives
        # separately from ``self.gammas`` so the SF predictors can train at
        # a shorter horizon (γ_sf, default 0.97) while the actor / Q_I
        # multigamma machinery uses ``self.gammas`` (length G_q, with γ_i
        # = ``gamma`` in the last slot). See ``TDJepa.compute_losses`` —
        # SF actions are always sliced to ``[..., -1:, :]`` (the γ_i
        # actor head, analogous to the α=0 actor pin) but the discount
        # in the SF Bellman backup is γ_sf.
        self.sf_gamma = sf_gamma
        self.sf_gammas = torch.Tensor([sf_gamma]).float()
        self.tau = tau
        self.reward_multiplier = reward_multiplier
        self.fake_filter = fake_filter
        self.num_actions_for_value_in_critic_loss = num_actions_for_value_in_critic_loss
        self.num_actions_for_value_in_actor_loss = num_actions_for_value_in_actor_loss
        self.fbc_filter_func = fbc_filter_func
        self.offline_coeff = offline_coeff
        self.online_coeff = online_coeff
        self.sf_loss_weight = sf_loss_weight
        self.use_target_actor = use_target_actor
        self.num_critics = num_critics
        self.num_critics_td = num_critics_td
        self.pass_obs_keys_to_actor = pass_obs_keys_to_actor or []
        # Paper-scale orthogonality coefficient, comparable to TD-JEPA's
        # `phi_ortho_coef` / `psi_ortho_coef`. Rescaled by 1/2 into
        # ``self._effective_fb_ortho_coeff`` to compensate for the explicit
        # 0.5 prefactor in our TD reduction (see TDJepa.__init__).
        self.fb_ortho_coeff = fb_ortho_coeff
        self._effective_fb_ortho_coeff = fb_ortho_coeff / 2.0
        self.fb_max_pairs = fb_max_pairs
        assert ortho_mode in (
            "all",
            "cross_traj",
            "sampled",
        ), f"Invalid ortho_mode: {ortho_mode}"
        self.ortho_mode = ortho_mode
        # mlp_seq_len / ac_seq_len validated and stashed before super().__init__()
        # `inference_mode` is set by the TDJepa subclass, which makes
        # it a required gin binding. The base class only declares the attribute
        # so static analysis and type inspection see it.
        self.inference_mode: InferenceMode

        self._inference_z: Optional[torch.Tensor] = None
        self._inference_alpha: Optional[torch.Tensor] = None
        # β mirrors α: per-actor beta tensor used at rollout/eval time.  Set
        # by ``propose_beta_for_collection`` (training) or by the eval phase
        # plumbing in BilevelExperiment._run_eval_phase.  ``None`` is
        # equivalent to a ConstantSchedule(ConstantAlpha(0.0)).
        self._inference_beta: Optional[torch.Tensor] = None
        self._last_traj_emb_b: Optional[torch.Tensor] = None
        self._z_cache_size = 8192
        self._z_cache: Optional[torch.Tensor] = None  # (cache_size, z_dim) on CPU
        self._z_cache_ptr = 0
        self._z_cache_full = False

        # Intrinsic curiosity critic stack (DVFB-style)
        self.alpha_dist: AlphaDistribution = alpha_dist or ConstantAlpha(0.0)
        # Evaluation always runs through an alpha schedule.  Callers that want
        # "no intrinsic reward" can simply leave this unset, yielding a constant
        # alpha=0 schedule (equivalent to ConstantSchedule(ConstantAlpha(0.0))).
        self.val_alpha_schedule: AlphaSchedule = (
            val_alpha_schedule or ConstantSchedule()
        )
        # β reuses the AlphaDistribution / AlphaSchedule classes verbatim
        # (BetaDistribution / BetaSchedule are module-scope aliases).  Default
        # of ConstantAlpha(0.0) keeps the extrinsic-reward branch fully off
        # for non-β baselines.
        self.beta_dist: BetaDistribution = beta_dist or ConstantAlpha(0.0)
        self.val_beta_schedule: BetaSchedule = val_beta_schedule or ConstantSchedule()
        self.intrinsic_critic_loss_weight = intrinsic_critic_loss_weight
        self.intrinsic_stack: Optional[IntrinsicCriticStack] = None

        # For the full αβ baseline (n_alphas > 1 AND n_betas > 1), pin the
        # first ``round(n_actors * frac)`` actors of every collection round
        # to (α=1, β=1) — the AMAGO-equivalent "pure extrinsic Q_P, no SF
        # blend, no intrinsic reward" corner.  No-op for presets where α or
        # β is constant.  Default 0.0 disables the pin.
        assert 0.0 <= pin_pure_extrinsic_fraction <= 1.0, (
            f"pin_pure_extrinsic_fraction must be in [0, 1], got "
            f"{pin_pure_extrinsic_fraction}"
        )
        self.pin_pure_extrinsic_fraction = pin_pure_extrinsic_fraction

    def _init_intrinsic_stack(
        self,
        obs_space: gym.spaces.Dict,
        rl2_space: gym.spaces.Box,
        action_space: gym.spaces.Space,
        max_seq_len: int,
    ):
        """Create the intrinsic critic stack. Must be called after self.actor and
        self.traj_encoder_pi exist (the IC reuses ``traj_encoder_pi.emb_dim``
        as its input ``policy_ctx_dim``).

        Constructs the stack iff α can be > 0 OR β can be > 0 — both axes
        feed Q_I (β controls the extrinsic-reward mix into Q_I's TD target,
        and at α=1, β=1 the stack is the only critic in use).
        """
        if self.alpha_dist.max_value > 0 or self.beta_dist.max_value > 0:
            with gin.config_scope("intrinsic"):
                self.intrinsic_stack = IntrinsicCriticStack(
                    obs_space=obs_space,
                    rl2_space=rl2_space,
                    action_space=action_space,
                    max_seq_len=max_seq_len,
                    gammas=self.gammas,
                    discrete=self.discrete,
                    action_dim=self.action_dim,
                    actor=self.actor,
                    policy_ctx_dim=self.traj_encoder_pi.emb_dim,
                    z_dim=self.psi_dim,
                    alpha_beta_enc_dim=self.alpha_beta_enc_dim,
                    ext_ctx_dim=self.ext_ctx_dim,
                    extrinsic_reward_multiplier=self.extrinsic_reward_multiplier,
                    zero_policy_ctx=self.zero_policy_ctx,
                    n_alphas=self.n_alphas,
                    n_betas=self.n_betas,
                    compute_online_q=(self.online_coeff > 0),
                )

    def _sample_alpha(self, B: int, device: torch.device) -> torch.Tensor:
        """Sample per-batch alpha via self.alpha_dist. Returns (B, 1)."""
        return self.alpha_dist.sample(B, device)

    def _sample_beta(self, B: int, device: torch.device) -> torch.Tensor:
        """Sample per-batch beta via self.beta_dist. Returns (B, 1)."""
        return self.beta_dist.sample(B, device)

    def _full_ab_active(self) -> bool:
        """True iff both α and β are sampled from non-constant distributions
        (i.e., the metabfm_alphabeta preset). Gates the
        ``pin_pure_extrinsic_fraction`` rollout pin. ``n_alphas`` /
        ``n_betas`` are TDJepa attributes; ``getattr`` keeps the helper
        safe to call before the subclass init has run.
        """
        return getattr(self, "n_alphas", 1) > 1 and getattr(self, "n_betas", 1) > 1

    def _pin_count(self, n_actors: int) -> int:
        """Number of actors deterministically forced to (α=1, β=1) per
        collection round. Only nonzero when full αβ is active AND the
        pin fraction is set; otherwise returns 0.
        """
        if not self._full_ab_active() or self.pin_pure_extrinsic_fraction <= 0.0:
            return 0
        return min(int(round(n_actors * self.pin_pure_extrinsic_fraction)), n_actors)

    def init_encoders(self) -> None:
        """Build primary + secondary backbone encoders.

        Overrides BaseAgent.init_encoders() to:
          - Pass specify_obs_keys directly to each tstep encoder
          - Build the secondary backbone (network B / psi)
        """
        self.tstep_encoder = self.tstep_encoder_type(
            obs_space=self.obs_space,
            rl2_space=self.rl2_space,
            specify_obs_keys=self._specify_obs_keys,
        )

        self.traj_encoder = self.traj_encoder_type(
            tstep_dim=self.tstep_encoder.emb_dim,
            max_seq_len=self.ac_seq_len,
        )

        self.tstep_encoder_b = self._tstep_encoder_b_type(
            obs_space=self.obs_space,
            rl2_space=self.rl2_space,
            specify_obs_keys=self._specify_obs_keys_b,
        )
        self.tstep_encoder_b.hide_rl2s = True
        self.traj_encoder_b = self._traj_encoder_b_type(
            tstep_dim=self.tstep_encoder_b.emb_dim,
            max_seq_len=self.ac_seq_len,
        )

    def init_hidden_state(self, batch_size: int, device: torch.device):
        """Initialize hidden states for both primary and secondary traj encoders."""
        return (
            self.traj_encoder.init_hidden_state(batch_size, device),
            self.traj_encoder_b.init_hidden_state(batch_size, device),
        )

    def reset_hidden_state(self, hidden_state, dones: np.ndarray):
        """Reset hidden states for both traj encoders on episode boundaries."""
        if hidden_state is None:
            return None
        h_a, h_b = hidden_state
        return (
            self.traj_encoder.reset_hidden_state(h_a, dones),
            self.traj_encoder_b.reset_hidden_state(h_b, dones),
        )

    @property
    @abc.abstractmethod
    def task_vector_dim(self) -> int:
        """Dimensionality of the z task vector (and the phi feature space)."""

    def _cache_z(self, z: torch.Tensor) -> None:
        """Store z vectors (B, z_dim) into a pre-allocated ring buffer.

        Buffer lives on the same device as ``z`` (typically GPU) to avoid a
        blocking device sync every training batch.
        """
        B = z.shape[0]
        z_detached = z.detach()
        if self._z_cache is None:
            self._z_cache = torch.empty(
                self._z_cache_size, z.shape[1], device=z_detached.device
            )
        end = self._z_cache_ptr + B
        if end <= self._z_cache_size:
            self._z_cache[self._z_cache_ptr : end] = z_detached
        else:
            first = self._z_cache_size - self._z_cache_ptr
            self._z_cache[self._z_cache_ptr :] = z_detached[:first]
            self._z_cache[: B - first] = z_detached[first:]
            self._z_cache_full = True
        self._z_cache_ptr = end % self._z_cache_size
        if end >= self._z_cache_size:
            self._z_cache_full = True

    def propose_z_for_collection(
        self, n_actors: int, device: torch.device
    ) -> torch.Tensor:
        """Propose z vectors for online training collection. Returns (n_actors, z_dim)."""
        if self._z_cache is None:
            return self._sample_z(n_actors, device)
        n_valid = self._z_cache_size if self._z_cache_full else self._z_cache_ptr
        cache_device = self._z_cache.device
        indices = torch.randint(n_valid, (n_actors,), device=cache_device)
        return self._z_cache[indices].to(device)

    def propose_alpha_for_collection(
        self, n_actors: int, device: torch.device
    ) -> torch.Tensor:
        """Propose per-actor alpha for online collection. Returns (n_actors, 1).

        When ``pin_pure_extrinsic_fraction > 0`` AND full αβ is active
        (``_full_ab_active() == True``), the first ``_pin_count(n_actors)``
        entries are deterministically set to 1.0. The complementary β
        side mirrors this in ``propose_beta_for_collection``, so the
        same actor indices are pinned to (α=1, β=1) on both axes.
        """
        alpha = self._sample_alpha(n_actors, device)
        n_pin = self._pin_count(n_actors)
        if n_pin > 0:
            alpha[:n_pin] = 1.0
        return alpha

    def propose_beta_for_collection(
        self, n_actors: int, device: torch.device
    ) -> torch.Tensor:
        """Propose per-actor beta for online collection. Returns (n_actors, 1).

        Mirrors ``propose_alpha_for_collection``: pins the same prefix of
        actor indices to β=1 when full αβ is active and the pin fraction
        is set. Combined with the α-side pin, the first ``n_pin`` actors
        roll out at the AMAGO-faithful (α=1, β=1) corner for the entire
        meta-trial.
        """
        beta = self._sample_beta(n_actors, device)
        n_pin = self._pin_count(n_actors)
        if n_pin > 0:
            beta[:n_pin] = 1.0
        return beta

    def _subsample_fb_indices(
        self, BL: int, mask_flat: torch.Tensor, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Return random index subset if fb_max_pairs > 0 and BL exceeds it."""
        if self.fb_max_pairs <= 0 or BL <= self.fb_max_pairs:
            return None
        valid_idx = mask_flat.nonzero(as_tuple=True)[0]
        n_valid = valid_idx.shape[0]
        n_keep = min(self.fb_max_pairs, n_valid)
        perm = torch.randperm(n_valid, device=device)[:n_keep]
        return valid_idx[perm]

    def _subsample_window(
        self,
        batch: Batch,
        full_reps: dict[str, torch.Tensor],
    ) -> Tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        int,
    ]:
        """Subsample a contiguous window for MLP loss terms.

        Args:
            batch: Full batch.
            full_reps: Dict of name -> (B, L_full, D) tensors from traj encoders.

        Returns:
            (reps, actions, dones, rl2s, rews, obs_window, time_idxs, L, gather_fn)
            where L is the window length.  ``gather_fn(src, length='L')`` can be
            used by the caller to window additional tensors (``length='L'`` uses
            ``idx_J``, ``length='L-1'`` uses ``idx_Jm1``).  When no subsampling
            occurs, ``gather_fn`` is the identity.
        """
        B, L_full = next(iter(full_reps.values())).shape[:2]
        J = self.mlp_seq_len
        if J is None or J >= L_full:
            _identity = lambda src, length="L": src
            return (
                full_reps,
                batch.actions,
                batch.dones,
                batch.rl2s,
                batch.rews,
                {k: batch.obs[k] for k in self.pass_obs_keys_to_actor},
                batch.time_idxs,
                L_full,
                _identity,
            )
        dev = next(iter(full_reps.values())).device
        valid = (~(batch.rl2s == self.pad_val).all(-1)).float()  # (B, L_full)
        valid_len = valid.sum(dim=1).long()
        max_start = (valid_len - J).clamp(min=0)
        center = (torch.rand(B, device=dev) * valid_len.float().clamp(min=1)).long()
        t0 = (center - J // 2).clamp(min=0).clamp(max=max_start)
        idx_J = t0.unsqueeze(1) + torch.arange(J, device=dev).unsqueeze(0)
        idx_Jm1 = t0.unsqueeze(1) + torch.arange(J - 1, device=dev).unsqueeze(0)

        def _gather(src, idx):
            if src.ndim == 2:
                return torch.gather(src, 1, idx)
            ix = idx
            for _ in range(src.ndim - 2):
                ix = ix.unsqueeze(-1)
            return torch.gather(src, 1, ix.expand(*idx.shape, *src.shape[2:]))

        def gather_fn(src, length="L"):
            idx = idx_J if length == "L" else idx_Jm1
            return _gather(src, idx)

        reps = {k: _gather(v, idx_J) for k, v in full_reps.items()}
        actions = _gather(batch.actions, idx_Jm1)
        dones = _gather(batch.dones, idx_Jm1)
        rews = _gather(batch.rews, idx_Jm1)
        rl2s = _gather(batch.rl2s, idx_J)
        time_idxs = _gather(batch.time_idxs, idx_J)
        obs_window = {
            k: _gather(batch.obs[k], idx_J) for k in self.pass_obs_keys_to_actor
        }
        return reps, actions, dones, rl2s, rews, obs_window, time_idxs, J, gather_fn

    def _ema_copy(self, target, online, tau: Optional[float] = None):
        """Override BaseAgent's per-parameter EMA loop with a fused foreach.

        See :func:`_polyak_ema_inplace` for the contract.
        """
        if tau is None:
            tau = self.tau
        _polyak_ema_inplace(target, online, tau)

    def _full_copy(self, target, online):
        """Override BaseAgent's per-parameter hard-copy loop with foreach.

        See :func:`_full_copy_inplace`.
        """
        _full_copy_inplace(target, online)

    def forward(self, batch: Batch, log_step: bool) -> torch.Tensor:
        self.update_info = {}
        sf_loss, actor_loss, intrinsic_critic_loss = self.compute_losses(
            batch, log_step
        )
        total = actor_loss + self.sf_loss_weight * sf_loss
        if intrinsic_critic_loss is not None:
            total = total + self.intrinsic_critic_loss_weight * intrinsic_critic_loss
        self.update_info["SF Loss"] = sf_loss
        self.update_info["Actor Loss"] = actor_loss
        if intrinsic_critic_loss is not None:
            self.update_info["Intrinsic Critic Loss"] = intrinsic_critic_loss
        if log_step:
            # Diagnostic only: mean raw env reward across valid (non-pad,
            # non-NO_REWARD) timesteps of the sampled batch. ZSRL never
            # consumes batch.rews directly for its TD targets (those are
            # intrinsic), so this doubles as a "is the agent actually
            # succeeding on the task" indicator. Mirrors AMAGO ``Agent``'s
            # ``Mean Reward (in training sequences)`` log field.
            rews_flat = batch.rews.squeeze(-1)  # (B, L-1)
            valid = (~(batch.rl2s == self.pad_val).all(-1))[:, 1:]
            valid = valid & (rews_flat != NO_REWARD)
            if valid.any():
                self.update_info["Mean Reward (in training sequences)"] = rews_flat[
                    valid
                ].mean()
        return total

    @abc.abstractmethod
    def compute_losses(
        self, batch: Batch, log_step: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def compute_intrinsic_reward(
        self, batch: Batch, log_step: bool
    ) -> Optional[torch.Tensor]:
        """Compute the intrinsic reward signal at the full batch sequence length.

        Runs BEFORE any ac_seq_len subsampling so episodic accumulators
        (E3B, NGU) see the full meta-trial.

        Writes two keys into ``batch.obs``:

        * ``"intrinsic_reward"`` — policy-visible reward (no RND).  Fed to
          ``traj_encoder_pi`` at training and rollout.
        * ``"intrinsic_reward_td"`` — full reward (with RND).  Consumed only
          by ``IntrinsicCriticStack.forward`` for the TD target.

        Returns an optional differentiable predictor loss (e.g. RND) that the
        caller should add to the total optimized loss, scaled by
        ``intrinsic_critic_loss_weight``.
        """
        if not hasattr(self, "update_info"):
            self.update_info = {}
        if self.intrinsic_stack is not None:
            ir_td, ir_obs, ir_loss = self.intrinsic_stack.compute_intrinsic_reward(
                batch,
                log_step=log_step,
            )
            batch.obs["intrinsic_reward"] = sanitize_published(
                "intrinsic_reward",
                ir_obs,
                log_step=log_step,
                update_info=self.update_info,
            )
            batch.obs["intrinsic_reward_td"] = sanitize_published(
                "intrinsic_reward_td",
                ir_td,
                log_step=log_step,
                update_info=self.update_info,
            )
            return self.intrinsic_critic_loss_weight * ir_loss
        if "rldp_error" in batch.obs:
            fallback = batch.obs["rldp_error"].detach()
            batch.obs["intrinsic_reward"] = fallback
            batch.obs["intrinsic_reward_td"] = fallback
        return None

    def slice_batch(self, batch: Batch) -> Batch:
        """Optionally slice the batch down from max_seq_len to ac_seq_len.

        Patterned on ``_subsample_window`` but runs BEFORE any encoder passes
        and therefore gathers EVERY ``batch.obs`` key (not just
        ``pass_obs_keys_to_actor``) so downstream §1 backbones see a coherent
        sliced batch.  Window start ``t0`` is sampled per-row as a uniform
        random contiguous valid window (mirrors ``_subsample_window``).

        No-op when ``ac_seq_len >= L_full``.

        Called by BilevelAgent.forward AFTER ``compute_intrinsic_reward`` so
        the RL agent sees a shorter batch while the intrinsic-reward
        accumulators see the full meta-trial.
        """
        J = self.ac_seq_len
        L_full = batch.rl2s.shape[1]
        if J >= L_full:
            return batch
        B = batch.rl2s.shape[0]
        dev = batch.rl2s.device
        valid = (~(batch.rl2s == self.pad_val).all(-1)).float()  # (B, L_full)
        valid_len = valid.sum(dim=1).long()
        max_start = (valid_len - J).clamp(min=0)
        center = (torch.rand(B, device=dev) * valid_len.float().clamp(min=1)).long()
        t0 = (center - J // 2).clamp(min=0).clamp(max=max_start)
        idx_J = t0.unsqueeze(1) + torch.arange(J, device=dev).unsqueeze(0)
        idx_Jm1 = t0.unsqueeze(1) + torch.arange(J - 1, device=dev).unsqueeze(0)

        def _gather(src, idx):
            if src.ndim == 2:
                return torch.gather(src, 1, idx)
            ix = idx
            for _ in range(src.ndim - 2):
                ix = ix.unsqueeze(-1)
            return torch.gather(src, 1, ix.expand(*idx.shape, *src.shape[2:]))

        new_obs = {k: _gather(v, idx_J) for k, v in batch.obs.items()}
        return Batch(
            obs=new_obs,
            rl2s=_gather(batch.rl2s, idx_J),
            rews=_gather(batch.rews, idx_Jm1),
            dones=_gather(batch.dones, idx_Jm1),
            actions=_gather(batch.actions, idx_Jm1),
            time_idxs=_gather(batch.time_idxs, idx_J),
        )

    @property
    def inference_z(self) -> Optional[torch.Tensor]:
        return self._inference_z

    @abc.abstractmethod
    def normalize_z(self, z_unnorm: torch.Tensor) -> torch.Tensor:
        """Project unnormalized task vector into the agent's canonical z space.

        TDJepa: psi_encoder.out_norm.
        """

    def extract_phi_for_blr(
        self, actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract phi(s) features from the last get_actions call for BLR updates.

        Returns (B, phi_dim).  Subclasses override (TDJepa: psi_encoder on
        the cached _last_traj_emb_b).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement extract_phi_for_blr"
        )

    def clip_param_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """Partition trainable params into clip groups for grouped grad clipping.

        Subclasses with structurally separable subsystems (TDJepa: AMAGO-side
        actor + IC vs SF-core phi/psi; RND as its own group) override this to
        return {"amago": [...], "tdjepa_core": [...], "rnd": [...]}.

        Default fallback puts everything into ``tdjepa_core`` so a single
        clip applies uniformly.
        """
        return {
            "amago": [],
            "tdjepa_core": list(self.trainable_params),
            "rnd": [],
        }

    def _policy_stats(self, mask, a_dist) -> dict:
        """Log actor distribution statistics (test-time gamma)."""
        mask = mask[..., -1, :]
        if self.discrete:
            entropy = a_dist.entropy().unsqueeze(-1)[..., -1, :]
            low_prob = torch.min(a_dist.probs, dim=-1, keepdims=True).values[..., -1, :]
            high_prob = torch.max(a_dist.probs, dim=-1, keepdims=True).values[
                ..., -1, :
            ]
            return {
                "Policy Per-timestep Entropy (test-time gamma)": utils.masked_avg(
                    entropy, mask
                ),
                "Policy Per-timstep Low Prob. (test-time gamma)": utils.masked_avg(
                    low_prob, mask
                ),
                "Policy Per-timestep High Prob. (test-time gamma)": utils.masked_avg(
                    high_prob, mask
                ),
                "Policy Overall Highest Prob.": (mask * a_dist.probs[..., -1, :]).max(),
            }
        else:
            entropy = -a_dist.log_prob(a_dist.sample()).sum(-1, keepdims=True)[
                ..., -1, :
            ]
            return {"Policy Entropy (test-time gamma)": utils.masked_avg(entropy, mask)}

    def _filter_stats(self, mask, logp_a, filter_) -> dict:
        mask = mask[:, :-1, ...]
        binary_filt = filter_ > 0
        masked_logp_a = logp_a[mask.bool()]
        return {
            "Minimum Action Logprob": masked_logp_a.min(),
            "Maximum Action Logprob": masked_logp_a.max(),
            "Mean Action Logprob": masked_logp_a.mean(),
            "Filter Max": filter_.max(),
            "Filter Min": filter_.min(),
            "Filter Mean": (mask * filter_).sum() / mask.sum(),
            "Filter 95th Percentile": torch.quantile(filter_.float(), 0.95),
            "Filter 75th Percentile": torch.quantile(filter_.float(), 0.75),
            "Pct. of Actions Approved by Binary FBC Filter (All Gammas)": utils.masked_avg(
                binary_filt, mask
            )
            * 100.0,
        }

    def _ortho_loss(
        self,
        key_name: str,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_flat: torch.Tensor,
        log_dict: dict = None,
    ) -> torch.Tensor:
        B, L_seq, _d = x.shape

        if self.ortho_mode == "sampled":
            valid_mask = mask_flat.reshape(B, L_seq)
            valid_counts = valid_mask.float().sum(dim=1)
            has_valid = valid_counts > 0
            safe_counts = valid_counts.clamp(min=1)
            rand_k = (torch.rand(B, device=x.device) * safe_counts).long()
            cumvalid = valid_mask.float().cumsum(dim=1)
            target = (rand_k + 1).float().unsqueeze(1)
            chosen = (cumvalid >= target).float().argmax(dim=1)
            x_sampled = x[torch.arange(B, device=x.device), chosen]
            cov = x_sampled @ x_sampled.T
            I = torch.eye(B, device=x.device, dtype=torch.bool)
            valid_pair = has_valid.unsqueeze(0) & has_valid.unsqueeze(1)
            cov_diag = cov.diag()[has_valid]
            cov_off_diag = cov[~I & valid_pair].pow(2)
        else:
            x = rearrange(x, "b l d -> (b l) d")
            sub_idx = self._subsample_fb_indices(x.shape[0], mask_flat, x.device)
            if sub_idx is not None:
                x = x[sub_idx]
                N = sub_idx.shape[0]
                cov = torch.matmul(x, x.T)
                I = torch.eye(N, device=x.device, dtype=torch.bool)
                cov_diag = cov.diag()
                if self.ortho_mode == "cross_traj":
                    traj_id = sub_idx // L_seq
                    same_traj = traj_id.unsqueeze(0) == traj_id.unsqueeze(1)
                    cov_off_diag = cov[~I & ~same_traj].pow(2)
                else:
                    cov_off_diag = cov[~I].pow(2)
            else:
                cov = torch.matmul(x, x.T)
                I = torch.eye(x.shape[0], device=x.device, dtype=torch.bool)
                cov_diag = cov.diag()[..., mask_flat]
                if self.ortho_mode == "cross_traj":
                    traj_id = torch.arange(x.shape[0], device=x.device) // L_seq
                    same_traj = traj_id.unsqueeze(0) == traj_id.unsqueeze(1)
                    cov_off_diag = cov[~I & ~same_traj & mask].pow(2)
                else:
                    cov_off_diag = cov[..., ~I & mask].pow(2)

        cov_diag_loss = -cov_diag.mean()
        cov_off_diag_loss = cov_off_diag.mean()
        ortho_loss = cov_diag_loss + 0.5 * cov_off_diag_loss
        if log_dict is not None:
            log_dict[f"Jepa/{key_name}/Orthogonal Loss Diagonal"] = (
                cov_diag_loss.detach()
            )
            log_dict[f"Jepa/{key_name}/Orthogonal Diagonal Min"] = cov_diag.min().item()
            log_dict[f"Jepa/{key_name}/Orthogonal Diagonal Max"] = cov_diag.max().item()
            log_dict[f"Jepa/{key_name}/Orthogonal Loss Off-Diagonal"] = (
                cov_off_diag_loss.detach()
            )
        return ortho_loss


@gin.configurable
@register_agent("tdjepa")
class TDJepa(ZeroShotAgent):
    """TD-JEPA (https://arxiv.org/abs/2510.00739)"""

    def __init__(
        self,
        obs_space: gym.spaces.Dict,
        rl2_space: gym.spaces.Box,
        action_space: gym.spaces.Space,
        max_seq_len: int,
        tstep_encoder_type: Type[TstepEncoder] = gin.REQUIRED,
        traj_encoder_type: Type[TrajEncoder] = gin.REQUIRED,
        tstep_encoder_b_type: Type[TstepEncoder] = gin.REQUIRED,
        traj_encoder_b_type: Type[TrajEncoder] = gin.REQUIRED,
        traj_encoder_pi_type: Type[TrajEncoder] = gin.REQUIRED,
        specify_obs_keys: list[str] = gin.REQUIRED,
        specify_obs_keys_b: list[str] = gin.REQUIRED,
        mlp_seq_len: int = gin.REQUIRED,
        ac_seq_len: int = gin.REQUIRED,
        num_critics: int = 2,
        num_critics_td: int = 2,
        reward_multiplier: float = 10.0,
        online_coeff: float = 1.0,
        offline_coeff: float = 0.0,
        sf_loss_weight: float = 10.0,
        gamma: float = 0.98,
        sf_gamma: float = 0.97,
        tau: float = 0.002,
        num_actions_for_value_in_critic_loss: int = 1,
        num_actions_for_value_in_actor_loss: int = 1,
        fbc_filter_func: callable = binary_filter,
        use_multigamma: bool = False,
        phi_dim: int = 256,
        phi_encoder_hidden_dim: int = 256,
        phi_predictor_hidden_dim: int = 1024,
        phi_predictor_loss_type: str = "mse",
        phi_predictor_two_hot_range_a: float = 1.0,
        phi_predictor_two_hot_min: Optional[float] = None,
        phi_predictor_two_hot_max: Optional[float] = None,
        phi_predictor_two_hot_bins: int = 12,
        psi_dim: int = 200,
        psi_encoder_hidden_dim: int = 256,
        psi_predictor_hidden_dim: int = 1024,
        psi_predictor_loss_type: str = "mse",
        psi_predictor_two_hot_range_a: float = 1.0,
        psi_predictor_two_hot_min: Optional[float] = None,
        psi_predictor_two_hot_max: Optional[float] = None,
        psi_predictor_two_hot_bins: int = 12,
        actor_type: Type[BaseActorHead] = Actor,
        predictor_type: Type[JepaPredictor] = JepaPredictor,
        encoder_type: Type[JepaEncoder] = JepaEncoder,
        pass_obs_keys_to_actor: Optional[Iterable[str]] = None,
        fb_ortho_coeff: float = 1.0,
        z_goal_ratio: float = 0.33,
        z_reward_ratio: float = 0.0,
        z_reward_intermediate: bool = True,
        reward_regression_coeff: float = 0.0,
        detach_actor: bool = False,
        random_ensembling: bool = True,
        normalize_q: bool = False,
        inference_mode: InferenceMode = gin.REQUIRED,
        fb_max_pairs: int = 0,
        ortho_mode: str = "all",
        alpha_dist: AlphaDistribution = None,
        val_alpha_schedule: AlphaSchedule = None,
        beta_dist: BetaDistribution = None,
        val_beta_schedule: BetaSchedule = None,
        intrinsic_critic_loss_weight: float = 1.0,
        n_alphas: int = 1,
        n_betas: int = 1,
        extrinsic_reward_multiplier: float = 1.0,
        extrinsic_tstep_encoder_type: Optional[Type[TstepEncoder]] = None,
        extrinsic_traj_encoder_type: Optional[Type[TrajEncoder]] = None,
        fourier_n_freq: int = 16,
        zero_policy_ctx: bool = False,
        pin_pure_extrinsic_fraction: float = 0.0,
    ):
        assert online_coeff > 0 or offline_coeff > 0, (
            "Need at least one of online_coeff / offline_coeff > 0; "
            "otherwise the actor receives no learning signal."
        )
        super().__init__(
            obs_space=obs_space,
            rl2_space=rl2_space,
            action_space=action_space,
            max_seq_len=max_seq_len,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            tstep_encoder_b_type=tstep_encoder_b_type,
            traj_encoder_b_type=traj_encoder_b_type,
            specify_obs_keys=specify_obs_keys,
            specify_obs_keys_b=specify_obs_keys_b,
            mlp_seq_len=mlp_seq_len,
            ac_seq_len=ac_seq_len,
            num_critics=num_critics,
            num_critics_td=num_critics_td,
            reward_multiplier=reward_multiplier,
            online_coeff=online_coeff,
            offline_coeff=offline_coeff,
            sf_loss_weight=sf_loss_weight,
            gamma=gamma,
            sf_gamma=sf_gamma,
            tau=tau,
            fake_filter=False,
            num_actions_for_value_in_critic_loss=num_actions_for_value_in_critic_loss,
            num_actions_for_value_in_actor_loss=num_actions_for_value_in_actor_loss,
            fbc_filter_func=fbc_filter_func,
            use_multigamma=use_multigamma,
            pass_obs_keys_to_actor=pass_obs_keys_to_actor,
            fb_ortho_coeff=fb_ortho_coeff,
            fb_max_pairs=fb_max_pairs,
            ortho_mode=ortho_mode,
            alpha_dist=alpha_dist,
            val_alpha_schedule=val_alpha_schedule,
            beta_dist=beta_dist,
            val_beta_schedule=val_beta_schedule,
            intrinsic_critic_loss_weight=intrinsic_critic_loss_weight,
            pin_pure_extrinsic_fraction=pin_pure_extrinsic_fraction,
        )
        self.inference_mode = inference_mode
        self.n_alphas = n_alphas
        self.n_betas = n_betas
        self.fourier_n_freq = fourier_n_freq
        # Per-scalar Fourier dim = 2 * n_freq sin/cos channels + 1 raw scalar
        # tail.  α and β each contribute this many dims to the actor / Q_I
        # MLP input via :func:`metabfm.rl.conditioning.fourier_encode`.
        self.alpha_beta_enc_dim = 2 * (2 * fourier_n_freq + 1)
        self.extrinsic_reward_multiplier = extrinsic_reward_multiplier
        # Debug knob — zero out ``policy_ctx`` (the TDJepa pi traj-encoder
        # output) at every actor MLP input AND every Q_I critic MLP input,
        # so the actor/critic only see ``[z, α_enc, β_enc, gated_ext]``
        # (actor) and ``[α_enc, β_enc, gated_ext, z?]`` (critic).
        # Useful for forcing ``pure_extrinsic`` (α=β=1) to be a faithful
        # AMAGO baseline that is provably independent of TDJepa SF
        # features. TDJepa SF training still runs; only its propagation
        # into the actor/critic heads is suppressed.
        self.zero_policy_ctx = zero_policy_ctx
        self.psi_dim = psi_dim
        self.phi_dim = phi_dim
        # Convert paper-scale `fb_ortho_coeff` to the value actually applied
        # to the ortho loss in `compute_losses`. Our TD reduction carries a
        # 0.5 prefactor relative to TD-JEPA's:
        #   TD-JEPA: `(P-tgt).pow(2).sum(-1).mean()` = D * mean(e^2)
        #   ours:    `masked_avg(0.5 * (P-tgt).pow(2), critic_mask)`
        #            = 0.5 * D * mean(e^2)
        # (the trailing `1` in `critic_mask` only zeros invalid rows; it
        # does not average out the feature dim D). The ortho loss formula
        # (`-mean(diag) + 0.5 * mean(off^2)`) is the same, so dividing by 2
        # makes the effective ortho:TD ratio match the paper at any phi/psi
        # configuration.
        self._effective_fb_ortho_coeff = self.fb_ortho_coeff / 2.0
        self.z_goal_ratio = z_goal_ratio
        self.z_reward_ratio = z_reward_ratio
        self.z_reward_intermediate = z_reward_intermediate
        self.reward_regression_coeff = reward_regression_coeff
        self.detach_actor = detach_actor
        self.normalize_q = normalize_q
        if normalize_q:
            self.register_buffer("_q_mu", torch.zeros(1))
            self.register_buffer("_q_nu", torch.ones(1) * 100.0)
            self._q_beta = 5e-4
            self.register_buffer("_q_t", torch.ones(1))
        self.random_ensembling = random_ensembling

        # SF predictors run at a single gamma (γ_sf) regardless of the
        # actor / Q_I multigamma machinery.  Keeping G_sf=1 means the
        # heavy phi/psi/target predictors don't pay the multigamma
        # broadcast tax (G× state replication on the critic input)
        # while the cheap actor + Q_I heads still get the per-token
        # gradient amplification AMAGO's Multigammas trick provides.
        # See ``compute_losses`` for the action-slicing convention that
        # routes the deployment-head (γ_i) actor outputs to these SF
        # predictors.
        sf_predictor_kwargs = {
            "action_dim": self.action_dim,
            "discrete": self.discrete,
            "gammas": self.sf_gammas,
            "num_ensemble": num_critics,
            "z_dim": psi_dim,
        }
        phi_encoder_kwargs = {"state_dim": self.traj_encoder.emb_dim}
        psi_encoder_kwargs = {"state_dim": self.traj_encoder_b.emb_dim}

        # Per-predictor two-hot config.  Bin range is auto-derived from
        # ``self.sf_gammas`` via the typical-envelope formula
        # ``±a/(1−γ_sf)``, with optional explicit ``(min, max)`` overrides.
        # The range knobs are inert when ``loss_type="mse"`` (the default).
        self.phi_predictor_loss_type = phi_predictor_loss_type
        self.phi_predictor_two_hot_range_a = phi_predictor_two_hot_range_a
        self.phi_predictor_two_hot_min_override = phi_predictor_two_hot_min
        self.phi_predictor_two_hot_max_override = phi_predictor_two_hot_max
        self.phi_predictor_two_hot_bins = phi_predictor_two_hot_bins
        self.psi_predictor_loss_type = psi_predictor_loss_type
        self.psi_predictor_two_hot_range_a = psi_predictor_two_hot_range_a
        self.psi_predictor_two_hot_min_override = psi_predictor_two_hot_min
        self.psi_predictor_two_hot_max_override = psi_predictor_two_hot_max
        self.psi_predictor_two_hot_bins = psi_predictor_two_hot_bins
        (
            self.phi_predictor_two_hot_min_resolved,
            self.phi_predictor_two_hot_max_resolved,
        ) = self._resolve_two_hot_range(
            phi_predictor_two_hot_range_a,
            phi_predictor_two_hot_min,
            phi_predictor_two_hot_max,
        )
        (
            self.psi_predictor_two_hot_min_resolved,
            self.psi_predictor_two_hot_max_resolved,
        ) = self._resolve_two_hot_range(
            psi_predictor_two_hot_range_a,
            psi_predictor_two_hot_min,
            psi_predictor_two_hot_max,
        )
        phi_predictor_two_hot_kwargs = {
            "loss_type": phi_predictor_loss_type,
            "two_hot_min": self.phi_predictor_two_hot_min_resolved,
            "two_hot_max": self.phi_predictor_two_hot_max_resolved,
            "two_hot_bins": phi_predictor_two_hot_bins,
        }
        psi_predictor_two_hot_kwargs = {
            "loss_type": psi_predictor_loss_type,
            "two_hot_min": self.psi_predictor_two_hot_min_resolved,
            "two_hot_max": self.psi_predictor_two_hot_max_resolved,
            "two_hot_bins": psi_predictor_two_hot_bins,
        }

        self.phi_encoder = encoder_type(
            **phi_encoder_kwargs, emb_dim=phi_dim, d_hidden=phi_encoder_hidden_dim
        )
        self.target_phi_encoder = encoder_type(
            **phi_encoder_kwargs, emb_dim=phi_dim, d_hidden=phi_encoder_hidden_dim
        )
        self.psi_encoder = encoder_type(
            **psi_encoder_kwargs, emb_dim=psi_dim, d_hidden=psi_encoder_hidden_dim
        )
        self.target_psi_encoder = encoder_type(
            **psi_encoder_kwargs, emb_dim=psi_dim, d_hidden=psi_encoder_hidden_dim
        )
        self.phi_predictor = predictor_type(
            **sf_predictor_kwargs,
            **phi_predictor_two_hot_kwargs,
            state_dim=phi_dim,
            out_dim=psi_dim,
            d_hidden=phi_predictor_hidden_dim,
        )
        self.target_phi_predictor = predictor_type(
            **sf_predictor_kwargs,
            **phi_predictor_two_hot_kwargs,
            state_dim=phi_dim,
            out_dim=psi_dim,
            d_hidden=phi_predictor_hidden_dim,
        )
        self.maximized_phi_predictor = predictor_type(
            **sf_predictor_kwargs,
            **phi_predictor_two_hot_kwargs,
            state_dim=phi_dim,
            out_dim=psi_dim,
            d_hidden=phi_predictor_hidden_dim,
        )
        # T_psi: psi_dim x A x Z --> phi_dim
        self.psi_predictor = predictor_type(
            **sf_predictor_kwargs,
            **psi_predictor_two_hot_kwargs,
            state_dim=psi_dim,
            out_dim=phi_dim,
            d_hidden=psi_predictor_hidden_dim,
        )
        self.target_psi_predictor = predictor_type(
            **sf_predictor_kwargs,
            **psi_predictor_two_hot_kwargs,
            state_dim=psi_dim,
            out_dim=phi_dim,
            d_hidden=psi_predictor_hidden_dim,
        )
        # Policy-side tstep encoder: a dedicated learned projection of
        # (phi, intrinsic_reward, prev_action, [rldp, idm, observed_normalized,
        # rldp_error]) whose output feeds ``traj_encoder_pi``.  This is the
        # only learned per-step projection on the actor + intrinsic-critic
        # side (the IC consumes the resulting policy_ctx directly), so
        # gradients from both the actor loss and the intrinsic critic loss
        # flow back through these parameters.
        #
        # Which extras are concatenated is controlled by
        # ``PolicyTstepEncoder.specify_obs_keys`` (gin); ``phi`` is always
        # the first input.  See ``metabfm.rl.policy_tstep``.  Callers
        # wrap phi in ``detach_if`` to honor ``detach_actor``.
        self.tstep_encoder_pi = PolicyTstepEncoder(
            phi_dim=phi_dim,
            prev_action_dim=self.action_dim,
            rldp_dim=(
                int(obs_space["rldp"].shape[0]) if "rldp" in obs_space.spaces else 0
            ),
            idm_dim=int(obs_space["idm"].shape[0]) if "idm" in obs_space.spaces else 0,
            observed_normalized_dim=(
                int(obs_space["observed_normalized"].shape[0])
                if "observed_normalized" in obs_space.spaces
                else 0
            ),
            rldp_error_dim=(
                int(obs_space["rldp_error"].shape[0])
                if "rldp_error" in obs_space.spaces
                else 0
            ),
        )
        self.traj_encoder_pi = traj_encoder_pi_type(
            tstep_dim=self.tstep_encoder_pi.emb_dim,
            max_seq_len=self.ac_seq_len,
        )
        pi_dim = self.traj_encoder_pi.emb_dim

        # Extrinsic-reward branch (β-gated).  Construction is skipped when β
        # is permanently 0 (β_dist.max_value == 0), saving a transformer
        # pass.  When skipped, ext_ctx_dim = 0 but the actor's state_dim still
        # includes the full Fourier-encoded β slot, so the actor input
        # shape is identical across baselines.
        self.extrinsic_tstep_encoder = None
        self.extrinsic_traj_encoder = None
        if self.beta_dist.max_value > 0:
            if (
                extrinsic_tstep_encoder_type is None
                or extrinsic_traj_encoder_type is None
            ):
                raise ValueError(
                    "TDJepa: beta_dist.max_value > 0 but "
                    "extrinsic_{tstep,traj}_encoder_type is None.  Bind both "
                    "via gin (see configs/beta/varying.gin)."
                )
            with gin.config_scope("extrinsic"):
                self.extrinsic_tstep_encoder = extrinsic_tstep_encoder_type(
                    obs_space=obs_space,
                    rl2_space=rl2_space,
                )
                self.extrinsic_tstep_encoder.hide_rl2s = True
                self.extrinsic_traj_encoder = extrinsic_traj_encoder_type(
                    tstep_dim=self.extrinsic_tstep_encoder.emb_dim,
                    max_seq_len=self.ac_seq_len,
                )
        self.ext_ctx_dim = (
            self.extrinsic_traj_encoder.emb_dim
            if self.extrinsic_traj_encoder is not None
            else 0
        )

        self.actor = actor_type(
            state_dim=pi_dim + psi_dim + self.alpha_beta_enc_dim + self.ext_ctx_dim,
            action_dim=self.action_dim,
            discrete=self.discrete,
            gammas=self.gammas,
        )
        self.hard_sync_targets()
        self._init_intrinsic_stack(obs_space, rl2_space, action_space, max_seq_len)

    def _cat_actor_input(
        self,
        policy_ctx: torch.Tensor,
        z: torch.Tensor,
        alpha_enc: torch.Tensor,
        beta_enc: torch.Tensor,
        gated_ext: torch.Tensor,
    ) -> torch.Tensor:
        """Canonical actor MLP input: ``[policy_ctx, z, α_enc, β_enc, gated_ext]``.

        Every call site that feeds the actor MLP (``get_actions``,
        ``compute_losses`` baseline pass, ``compute_losses`` multi-α/β pass)
        MUST use this helper so the concat order stays in sync with
        ``actor.state_dim = pi_dim + psi_dim + alpha_beta_enc_dim + ext_ctx_dim``.

        When ``zero_policy_ctx`` is True we zero BOTH ``policy_ctx`` and
        ``z`` before the concat: ``z`` is the BLR posterior sample over
        psi-space and only carries task signal in conjunction with the
        SF-shaped policy_ctx stream, so leaving it as random noise would
        defeat the purpose of the AMAGO-faithful debug knob.
        """
        if self.zero_policy_ctx:
            policy_ctx = torch.zeros_like(policy_ctx)
            z = torch.zeros_like(z)
        return torch.cat([policy_ctx, z, alpha_enc, beta_enc, gated_ext], dim=-1)

    def _resolve_two_hot_range(
        self,
        range_a: float,
        min_override: Optional[float],
        max_override: Optional[float],
    ) -> Tuple[float, float]:
        """Resolve the per-coord two-hot bin range.

        Auto-derives the symmetric range ``±a/(1−γ_sf)`` (the typical-envelope
        formula for sphere-normalized SF targets), with optional explicit
        ``(min, max)`` overrides for cases where telemetry shows the auto
        range is too tight or too wide. ``γ_sf`` is read from
        ``self.sf_gammas`` (a single-element tensor).

        Args:
            range_a: dimensionless typical-envelope multiplier (the ``a`` in
                ``±a/(1−γ_sf)``). Default ``1.0``.
            min_override: if not ``None``, replaces the auto lower bound.
            max_override: if not ``None``, replaces the auto upper bound.

        Returns:
            ``(lo, hi)`` floats with ``lo < hi``.
        """
        sf_gamma = float(self.sf_gammas.max().item())
        assert (
            sf_gamma < 1.0
        ), f"sf_gamma must be < 1 for the typical-envelope formula; got {sf_gamma}"
        auto_v = float(range_a) / (1.0 - sf_gamma)
        lo = float(min_override) if min_override is not None else -auto_v
        hi = float(max_override) if max_override is not None else +auto_v
        assert lo < hi, (
            f"two_hot range must be non-empty: ({lo}, {hi}) "
            f"(range_a={range_a}, min_override={min_override}, max_override={max_override})"
        )
        return lo, hi

    def _ensemble_subset_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """(B, L, C, G, dim) -> (B, L, 1, G, dim) via random subset mean."""
        if not self.random_ensembling:
            return tensor.mean(2, keepdim=True)
        B, L, C, G, dim = tensor.shape
        random_subset = torch.randint(
            low=0, high=C, size=(B, L, self.num_critics_td, G, 1), device=tensor.device
        )
        random_subset = random_subset.expand(-1, -1, -1, -1, dim)
        subset = torch.take_along_dim(tensor, random_subset, dim=2)
        return subset.mean(2, keepdim=True)

    def compute_losses(
        self, batch: Batch, log_step: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # fmt: off
        active_log_dict = self.update_info if log_step else None
        detach_if = lambda t_: t_.detach() if self.detach_actor else t_

        # ═══════════════════════════════════════════════════════════════════
        # § 1.  Backbone forwards at L_full (dynamics-adjacent encoders)
        # ═══════════════════════════════════════════════════════════════════
        # tstep encoders (phi / psi backbones).
        o   = self.tstep_encoder  (obs=batch.obs, rl2s=batch.rl2s, log_dict=active_log_dict)  # (B, L_full, D_o)
        o_b = self.tstep_encoder_b(obs=batch.obs, rl2s=batch.rl2s)                             # (B, L_full, D_ob)
        B, L_full, _ = o.shape

        # traj encoders on the full sequence.
        s_rep_full,   _ = self.traj_encoder  (seq=o,   time_idxs=batch.time_idxs, hidden_state=None, log_dict=active_log_dict)  # (B, L_full, emb_a)
        s_rep_b_full, _ = self.traj_encoder_b(seq=o_b, time_idxs=batch.time_idxs, hidden_state=None)                            # (B, L_full, emb_b)
        assert s_rep_full  .shape == (B, L_full, self.traj_encoder  .emb_dim)
        assert s_rep_b_full.shape == (B, L_full, self.traj_encoder_b.emb_dim)

        # phi / psi encoders at full length.
        phi_full = self.phi_encoder(s_rep_full)    # (B, L_full, phi_dim)
        psi_full = self.psi_encoder(s_rep_b_full)  # (B, L_full, psi_dim)

        # target encoders give "next" reps shifted by +1 (no grad).
        with torch.no_grad():
            next_phi_full = self.target_phi_encoder(s_rep_full  [:, 1:, ...])  # (B, L_full-1, phi_dim)
            next_psi_full = self.target_psi_encoder(s_rep_b_full[:, 1:, ...])  # (B, L_full-1, psi_dim)

        # Masks at full length.
        full_valid_state = (~(batch.rl2s == self.pad_val).all(-1))                      # (B, L_full) bool
        state_mask_full  = full_valid_state[:, 1:].float().unsqueeze(-1)                # (B, L_full-1, 1)
        actor_mask_full  = F.pad(state_mask_full, (0, 0, 0, 1), "constant", 0.0)        # (B, L_full, 1)
        G = len(self.gammas)
        actor_mask_full  = repeat(actor_mask_full, f"b l 1 -> b l {G} 1")               # (B, L_full, G, 1)

        # ═══════════════════════════════════════════════════════════════════
        # § 2.  Actor + intrinsic critic at L_full
        # ═══════════════════════════════════════════════════════════════════
        C = len(self.phi_predictor)
        K_c = self.num_actions_for_value_in_critic_loss if not self.discrete else 1
        K_a = self.num_actions_for_value_in_actor_loss  if not self.discrete else 1

        # ── three-tier z sampling (uses full-length psi_full for reward-z BLR) ──
        z = self._sample_z(B, o.device)                                                 # (B, psi_dim)
        z_random = z.clone()
        has_reward = None
        use_goal = None
        z_unnorm_reward = None
        z_reward_norm = None
        valid_rews = None
        use_reward = torch.zeros(B, device=o.device, dtype=torch.bool)
        if isinstance(self.inference_mode, OnlineBLR):
            valid_rews = full_valid_state[:, 1:] & (batch.rews.squeeze(-1) != NO_REWARD)  # (B, L_full-1)
            has_reward = valid_rews.any(dim=1)                                            # (B,)
            if self.z_reward_ratio > 0:
                use_reward = has_reward & (torch.rand(B, device=o.device) < self.z_reward_ratio)
                if use_reward.any():
                    z_reward_norm, z_unnorm_reward = self._compute_reward_z(
                        psi_full, batch.rews, valid_rews, use_reward, B, L_full
                    )
                    z[use_reward] = z_reward_norm[use_reward]
            if log_step and z_unnorm_reward is None and has_reward.any():
                z_reward_norm, z_unnorm_reward = self._compute_reward_z(
                    psi_full, batch.rews, valid_rews, has_reward, B, L_full
                )

        z_goal = None
        if self.z_goal_ratio > 0:
            valid_counts = state_mask_full.squeeze(-1).sum(dim=1).clamp(min=1)
            random_valid_indices = (torch.rand(B, device=psi_full.device) * valid_counts).long().clamp(0, L_full - 1)
            random_valid_psi = psi_full[torch.arange(B, device=psi_full.device), random_valid_indices]
            shuffled_psi = random_valid_psi[torch.randperm(B, device=psi_full.device)]
            z_goal = shuffled_psi
            use_goal = (torch.rand(B, device=o.device) < self.z_goal_ratio) & ~use_reward
            if use_goal.any():
                z[use_goal] = shuffled_psi[use_goal]

        if log_step:
            n_reward_z   = int(use_reward.sum().item())
            n_goal_z     = int(use_goal  .sum().item()) if use_goal   is not None else 0
            n_has_reward = int(has_reward.sum().item()) if has_reward is not None else 0
            self.update_info.update({
                "Z/Reward Fraction":      n_reward_z   / B,
                "Z/Goal Fraction":        n_goal_z     / B,
                "Z/Random Fraction":      (B - n_reward_z - n_goal_z) / B,
                "Z/Has Reward Fraction":  n_has_reward / B,
            })

        self._cache_z(z)
        z_full = repeat(z, "b d -> b l d", l=L_full).detach()                            # (B, L_full, psi_dim)
        assert z_full.shape == (B, L_full, self.psi_dim)
        zeros_alpha_full = torch.zeros(B, L_full, 1, device=phi_full.device)             # (B, L_full, 1)
        # Pre-Fourier-encoded zero-α / zero-β tensors used by the baseline
        # actor pass and by the α≡0/β≡0 fallback below. Same
        # shape as the per-slot encoded tensors so the actor input dim is
        # invariant across baselines (equal to the gin-bound actor.state_dim).
        zeros_alpha_enc_full = fourier_encode(zeros_alpha_full, self.fourier_n_freq)     # (B, L_full, 2*n_freq+1)
        zeros_beta_enc_full  = zeros_alpha_enc_full                                       # α=β=0 ⇒ same encoding
        zeros_gated_ext_full = torch.zeros(
            B, L_full, self.ext_ctx_dim, device=phi_full.device,
        )                                                                                # (B, L_full, ext_ctx_dim)

        straight_from_obs_full = {k: batch.obs[k] for k in self.pass_obs_keys_to_actor}

        # ── Intrinsic reward (two channels) ──
        # ``intrinsic_reward`` (obs-visible, no RND) is fed to
        # tstep_encoder_pi; ``intrinsic_reward_td`` (with RND) is used only
        # as the TD-target reward for IntrinsicCriticStack.forward.  Both are
        # written by ZeroShotAgent.compute_intrinsic_reward before
        # slice_batch, so they match the current batch shape.
        intrinsic_reward_obs_full = batch.obs["intrinsic_reward"]                       # (B, L_full, 1)
        intrinsic_reward_td_full  = batch.obs.get(
            "intrinsic_reward_td", intrinsic_reward_obs_full
        )                                                                                # (B, L_full, 1)

        # ── policy tstep + trajectory encoders (identity or memoryful) ──
        # tstep_encoder_pi projects (phi, intrinsic_reward, prev_action) into a
        # per-step feature.  intrinsic_reward is detached at source (position
        # 0 is zero), so no gradient flows into the intrinsic critic / RND
        # via this channel.  detach_if on phi keeps the actor loss from
        # updating phi/psi when detach_actor=True, but tstep_encoder_pi's own
        # parameters DO receive actor-loss gradients.  a' is sampled by
        # shifting the online actor's outputs by one timestep (TD-JEPA's
        # π(ϕ(s'))), so no separate target-actor forward is needed.
        pi_obs_full = {**batch.obs, "intrinsic_reward": intrinsic_reward_obs_full}
        pi_input_full = self.tstep_encoder_pi(
            phi=detach_if(phi_full), obs=pi_obs_full,
        )                                                                                # (B, L_full, pi_tstep_dim)
        policy_ctx_full, _ = self.traj_encoder_pi(
            seq=pi_input_full, time_idxs=batch.time_idxs, hidden_state=None,
        )                                                                                # (B, L_full, pi_dim)

        # ── baseline-(α=0, β=0) actor pass (diagnostics + a' sampling) ──
        # Uses the same actor.state_dim as the per-slot multi-α actor pass:
        # ``cat(policy_ctx, z, fourier_encode(α=0), fourier_encode(β=0),
        # zeros(ext_ctx_dim))``.  Equivalent to the (slot=0,0) cell of the
        # multi-α pass when corner pinning is on.
        actor_base_inp = self._cat_actor_input(
            policy_ctx_full, z_full.detach(),
            zeros_alpha_enc_full, zeros_beta_enc_full, zeros_gated_ext_full,
        )
        a_dist = self.actor(
            actor_base_inp,
            straight_from_obs=straight_from_obs_full,
        )

        # ── a' via online actor shifted by one timestep (TDJepa paper's π(ϕ(s'))) ──
        # α is constant along L in each batch row, so α_t = α_{t+1}; the shift
        # preserves per-trial α correctly.
        with torch.no_grad():
            a_prime_dist = a_dist
            a_prime_full_nshift = self._sample_k_actions(a_prime_dist, k=K_c)
            a_prime_full = a_prime_full_nshift[:, :, 1:, ...].detach()                   # (K_c, B, L_full-1, G, A_dim)

        # ── multi-(α, β) cross-product expansion for actor + intrinsic critic ──
        # NA × NB grid of (α, β) slots per training step.  Slot (0, 0) /
        # (NA-1, 0) / (NA-1, NB-1) are pinned to (0, 0) / (α_max, 0) /
        # (α_max, β_max) for clean per-corner logging when both axes have
        # ≥ 2 slots.  For α≡0 or β≡0 baselines NA / NB collapse to 1 and
        # cross-product reduces to a single (zero-α, zero-β) slot.
        NA = self.n_alphas
        NB = self.n_betas
        NAB = NA * NB
        # has_intrinsic gates the multi-slot α/β cross-product expansion AND
        # the IC forward.  Triggered whenever the IC stack is constructed —
        # _init_intrinsic_stack only builds it if α or β can be > 0, so the
        # actor MUST see meaningful α/β values (not zeros) and Q_I MUST be
        # trained.  Specifically: pure_extrinsic pins NA=NB=1 with α=β=1,
        # so without this branch the actor would see fourier_encode(0) for
        # both axes and the IC forward would never run.
        has_intrinsic = self.intrinsic_stack is not None
        has_ext_branch = self.extrinsic_traj_encoder is not None
        rep_ab = lambda v: repeat(v, "b ... -> (a b) ...", a=NAB)

        if has_intrinsic:
            # Independent α per (slot, batch-row); same for β.
            alphas = self.alpha_dist.sample((NA, B), phi_full.device)                     # (NA, B, 1)
            betas  = self.beta_dist .sample((NB, B), phi_full.device)                     # (NB, B, 1)
            if NA >= 2:
                alphas[ 0, :, :] = 0.0
                alphas[-1, :, :] = self.alpha_dist.max_value
            if NB >= 2:
                betas [ 0, :, :] = 0.0
                betas [-1, :, :] = self.beta_dist.max_value
            # ── Mandatory β-gate on reward-blind trajectories ──
            # If every non-pad transition in a row is NO_REWARD, training
            # Q_I at β > 0 would feed (β · -128.0) into the TD target —
            # poison.  Force β=0 across all NB slots for those rows;
            # gated_ext for those (slot, row) cells then collapses to
            # zero downstream.  No-op unless the env writes NO_REWARD.
            # The per-corner pin at slot -1 is also gated to 0 on
            # reward-blind rows (intentional — there is no extrinsic
            # signal to pin to).
            _valid_rews_b   = full_valid_state[:, 1:] & (batch.rews.squeeze(-1) != NO_REWARD)
            _traj_has_rew_b = _valid_rews_b.any(dim=1)                                          # (B,) bool
            betas = betas * _traj_has_rew_b.view(1, B, 1).to(betas.dtype)
            if log_step:
                _n_valid_steps = full_valid_state[:, 1:].sum().clamp(min=1).float()
                self.update_info["Reward/Sequences With Reward (Frac)"]  = (
                    _traj_has_rew_b.float().mean().item()
                )
                self.update_info["Reward/Timesteps With Reward (Frac)"] = (
                    _valid_rews_b.sum().float() / _n_valid_steps
                ).item()
                self.update_info["Reward/Sequences Beta-Gated"] = float(
                    (B - int(_traj_has_rew_b.sum().item()))
                )
            # Cross-product to (NA, NB, B, 1).
            alphas_grid = alphas.unsqueeze(1).expand(NA, NB, B, 1).contiguous()
            betas_grid  = betas .unsqueeze(0).expand(NA, NB, B, 1).contiguous()
            policy_ctx_r       = rep_ab(policy_ctx_full)
            z_r                = rep_ab(z_full)
            alpha_seq_full     = repeat(alphas_grid, "a c b 1 -> (a c b) l 1", l=L_full)   # (NAB*B, L_full, 1)
            beta_seq_full      = repeat(betas_grid,  "a c b 1 -> (a c b) l 1", l=L_full)   # (NAB*B, L_full, 1)
            sfobs_r = {k: rep_ab(v) for k, v in straight_from_obs_full.items()} if straight_from_obs_full else {}
        else:
            alphas = None
            betas  = None
            alphas_grid = None
            betas_grid  = None
            policy_ctx_r       = policy_ctx_full
            z_r                = z_full
            alpha_seq_full     = zeros_alpha_full
            beta_seq_full      = zeros_alpha_full
            sfobs_r            = straight_from_obs_full
            NA = 1
            NB = 1
            NAB = 1

        # ── Extrinsic-reward branch (β-gated) ──
        # Run the extrinsic encoder ONCE per row (B), then repeat NAB times
        # and multiply by β_seq for the architectural β-gating.  At β=0 cells
        # gated_ext is identically zero regardless of ext_ctx contents (the
        # actor / Q_I cannot leak any extrinsic signal at β=0).
        if has_ext_branch:
            ext_tstep_full = self.extrinsic_tstep_encoder(
                obs=batch.obs, rl2s=batch.rl2s
            )                                                                              # (B, L_full, ext_tstep_dim)
            ext_ctx_full, _ = self.extrinsic_traj_encoder(
                seq=ext_tstep_full,
                time_idxs=batch.time_idxs,
                hidden_state=None,
            )                                                                              # (B, L_full, ext_ctx_dim)
            ext_ctx_r = rep_ab(ext_ctx_full)                                               # (NAB*B, L_full, ext_ctx_dim)
            gated_ext_full = beta_seq_full * ext_ctx_r                                     # β · ext_ctx (β=0 zeros it)
        else:
            gated_ext_full = torch.zeros(
                NAB * B if has_intrinsic else B, L_full, 0, device=phi_full.device,
            )                                                                              # 0-dim → no contribution

        # ── Fourier-encode α and β once per slot (reused for actor + Q_I) ──
        alpha_seq_enc_full = fourier_encode(alpha_seq_full, self.fourier_n_freq)           # (NAB*B, L_full, 2*nf+1)
        beta_seq_enc_full  = fourier_encode(beta_seq_full,  self.fourier_n_freq)

        # ── multi-(α, β) actor forward at (NAB*B, L_full) ──
        # NOTE: do NOT apply detach_if to policy_ctx_r here.  detach_actor's
        # intent is "don't let actor loss update the *dynamics backbone* (phi /
        # s_rep)"; that's already implemented by detach_if(phi_full) at the
        # traj_encoder_pi input above.  traj_encoder_pi is part of the actor's
        # own module — gradients from the actor loss MUST flow through
        # policy_ctx_r into traj_encoder_pi.parameters() so a memoryful pi
        # encoder (ff_policy_tformer) actually trains.
        actor_multi_inp = self._cat_actor_input(
            policy_ctx_r, z_r.detach(),
            alpha_seq_enc_full, beta_seq_enc_full, gated_ext_full,
        )
        a_dist_multi = self.actor(
            actor_multi_inp,
            straight_from_obs=sfobs_r,
        )
        a_agent_full = self._sample_k_actions(a_dist_multi, k=K_a)                        # (K_a, NAB*B, L_full, G, A_dim)

        # V_td / normalize_q EMA / V_combined / online_term are computed in §3
        # so the expensive maximized_phi_predictor call runs at L_w instead of
        # L_full.  a_agent_full stays at L_full for the intrinsic critic below;
        # §3 gathers it to the window so V_td uses matched action samples.

        # ── intrinsic critic @ (NAB*B, L_full) ──
        Q_P_raw_full_ab = None  # (NAB*B, L_full-1, C, G_q, 1) raw env-scale, full ensemble
        Q_P_norm_ab     = None  # (NAB*B, L_full-1, G_q, 1) popart-normalized, min over critics (online DPG)
        Q_buf_ic_ab     = None
        intrinsic_critic_loss = None
        if has_intrinsic:
            # Pass the already-rep_ab'd policy_ctx_r as the IC's state input;
            # batch supplies masks / actions / dones / rl2s.
            # a' via online multi-(α, β) actor shifted by one timestep.
            with torch.no_grad():
                a_prime_multi_full = self._sample_k_actions(a_dist_multi, k=K_c)
                a_prime_multi = a_prime_multi_full[:, :, 1:, ...].detach()                  # (K_c, NAB*B, L_full-1, G, A_dim)

            intrinsic_critic_loss, q_p_agent, q_p_buffer = self.intrinsic_stack(
                policy_ctx_r, batch, intrinsic_reward_td_full,
                a_agent_full, a_prime_multi,
                alpha_seq_enc_full, beta_seq_enc_full, beta_seq_full,
                gated_ext_full, z_r, log_step,
            )
            # q_p_agent / q_p_buffer are raw env-scale (scalar critic:
            # popart-denormalized; two-hot: bin_dist_to_raw_vals).
            # Keep the full-ensemble Q_P_raw and reduce two ways:
            #   - .min(C_axis) for the online DPG term (CDQ pessimism, as in
            #     AMAGO's Agent.forward).
            #   - .mean(C_axis) for the offline-advantage filter (as in
            #     AMAGO's MultiTaskAgent.forward).
            Q_P_raw_full_ab = q_p_agent[:, :-1, ...]                                       # (NAB*B, L_full-1, C, G_q, 1)
            Q_P_norm_ab     = self.intrinsic_stack.normalize_q(
                Q_P_raw_full_ab.min(2).values
            )                                                                              # (NAB*B, L_full-1, G_q, 1)
            Q_buf_ic_ab     = q_p_buffer                                                   # (NAB*B, L_full-1, C, G_q, 1) raw env-scale

        # ═══════════════════════════════════════════════════════════════════
        # § 3.  Windowed section: subsample to L_w, then compute BOTH the
        # actor loss (V_td / V_combined / online_term / offline_term) AND the
        # TDJepa critic TD loss over the same window.  Running V_td at L_w
        # keeps maximized_phi_predictor off the L_full path.  Q_P values from
        # the full-length intrinsic critic are sliced to the window via the
        # _gather_ab helper below.
        # ═══════════════════════════════════════════════════════════════════
        # Full-length tensors to window.  L_full-1 tensors (next_phi/next_psi,
        # a_prime, and any (A*B) multi-alpha tensors used by the offline term)
        # are gathered via gather_fn(..., length="L-1") after the call.
        full_tensors = {
            "s_rep":     s_rep_full,     # (B, L_full, emb_a)
            "s_rep_b":   s_rep_b_full,   # (B, L_full, emb_b)
            "phi":       phi_full,       # (B, L_full, phi_dim)
            "psi":       psi_full,       # (B, L_full, psi_dim)
        }
        reps, actions, dones, rl2s_w, rews, obs_window, _, L, gather_fn = (
            self._subsample_window(batch, full_tensors)
        )
        if log_step:
            self.update_info["Seq/MLP Window Length"] = L
        phi = reps["phi"]                                                               # (B, L, phi_dim)
        psi = reps["psi"]                                                               # (B, L, psi_dim)

        # Gather L_full-1 tensors onto the window (length = L-1 when windowed, == L_full-1 identity otherwise).
        next_phi_w = gather_fn(next_phi_full, "L-1")                                    # (B, L-1, phi_dim)
        next_psi_w = gather_fn(next_psi_full, "L-1")                                    # (B, L-1, psi_dim)

        # z repeated over the windowed length (constant per sequence).
        z_w = repeat(z, "b d -> b l d", l=L).detach()                                    # (B, L, psi_dim)

        # Action buffer on the window.
        a = actions.clamp(0, 1.0) if self.discrete else actions.clamp(-1.0, 1.0)        # (B, L-1, A_dim)
        assert a.shape[1] == L - 1
        a_buffer = F.pad(a, (0, 0, 0, 1), "replicate")                                  # (B, L, A_dim)
        a_buffer = repeat(a_buffer, f"b l a -> b l {G} a")                              # (B, L, G, A_dim)
        a_buffer = self.actor.policy_dist.action_from_buffer(a_buffer)

        # Windowed masks.
        # ``critic_mask`` carries the multigamma G_q axis and is used by Q_P
        # logging / popart updates inside the intrinsic stack; the SF
        # phi/psi loss runs at G_sf=1 so it gets its own collapsed mask
        # to match the reduced predictor output shape.
        state_mask_w   = (~((rl2s_w == self.pad_val).all(-1, keepdim=True))).float()[:, 1:, ...]  # (B, L-1, 1)
        actor_mask_w   = F.pad(state_mask_w, (0, 0, 0, 1), "constant", 0.0)                        # (B, L, 1)
        actor_mask_w   = repeat(actor_mask_w, f"b l 1 -> b l {G} 1")                               # (B, L, G_q, 1)
        critic_mask    = repeat(state_mask_w, f"b l 1 -> b l {C} {G} 1")                           # (B, L-1, C, G_q, 1)
        sf_critic_mask = repeat(state_mask_w, f"b l 1 -> b l {C} 1 1")                             # (B, L-1, C, G_sf=1, 1)
        fb_mask_flat   = state_mask_w.reshape(B * (L - 1)).bool()
        fb_mask        = fb_mask_flat[:, None] & fb_mask_flat[None, :]

        # ── Helper: gather a (NAB*B, L_full[-1], ...) tensor onto the window ──
        # Replicates per-row before windowing so the gather indexing aligns
        # with the original (B, L_full) batch axis, then re-flattens to
        # (NAB*B, L, ...) for downstream consumers.
        def _gather_ab(t: torch.Tensor, length: str) -> torch.Tensor:
            t_b = rearrange(t, "(a b) l ... -> b l a ...", a=NAB)
            t_w = gather_fn(t_b, length)
            return rearrange(t_w, "b l a ... -> (a b) l ...")

        # ── Multi-(α, β) inputs at L_w (for V_td) ──
        # phi / z are slot-independent → rep_ab windowed tensors.
        # α / β are constant over time → rebuild at L_w from the slot grids.
        # a_agent_full was sampled at L_full in §2 (for the intrinsic critic);
        # gather to the window so V_td uses matched action samples.
        phi_r_w     = rep_ab(phi)                                                                # (NAB*B, L, phi_dim)
        z_r_w       = rep_ab(z_w)                                                                # (NAB*B, L, psi_dim)
        if has_intrinsic:
            alpha_seq_w = repeat(alphas_grid, "a c b 1 -> (a c b) l 1", l=L)                     # (NAB*B, L, 1)
        else:
            alpha_seq_w = torch.zeros(B, L, 1, device=phi.device)                                # NAB == 1
        a_agent_w = rearrange(
            gather_fn(
                rearrange(a_agent_full, "k (a b) l g d -> b l a k g d", a=NAB),
                "L",
            ),
            "b l a k g d -> k (a b) l g d",
        )                                                                                         # (K_a, NAB*B, L, G, A_dim)

        # ── V_td via SF phi_predictor at (NAB*B, L) ──
        # SF predictor runs at G_sf=1; slice a_agent_w to the deployment-head
        # (γ_i = self.gammas[-1]) action slot before passing in.  Gradient
        # through V_td therefore only flows back through the γ_i actor
        # head; auxiliary multigamma heads see no SF anchor signal and
        # learn purely from -α·Q_P(g) (regularizes the deployment head
        # via parameter sharing).
        #
        # Mirrors the IC `compute_online_q` recycling pattern: when
        # online_coeff > 0, use the hard-synced `maximized_phi_predictor` so
        # the online DPG term has gradients.  When online_coeff == 0, use
        # `self.phi_predictor` under no_grad to skip the autograd graph.
        if self.online_coeff > 0:
            phi_agent_w = self.maximized_phi_predictor(
                phi_r_w.detach(), a_agent_w[:, :, :, -1:, :], z_r_w
            )[:, :, :-1, ...]
        else:
            with torch.no_grad():
                phi_agent_w = self.phi_predictor(
                    phi_r_w.detach(), a_agent_w[:, :, :, -1:, :], z_r_w
                )[:, :, :-1, ...]
        # (K_a, NAB*B, L-1, C, G_sf=1, psi_dim)
        V_td_ab     = torch.einsum(
            "kblcgd,bld->kblcg", phi_agent_w, z_r_w[:, :-1, ...]
        ).unsqueeze(-1).mean(0)                                                                    # (NAB*B, L-1, C, G, 1)
        V_td_min_ab = V_td_ab.min(2).values                                                        # (NAB*B, L-1, G, 1)

        # normalize_q EMA update (on L_w).
        if self.normalize_q:
            with torch.no_grad():
                nmask = rep_ab(actor_mask_w[:, :-1, ...])                                          # (NAB*B, L-1, G, 1)
                total = nmask.sum().clamp(min=1)
                masked_mean    = (V_td_min_ab       * nmask).sum() / total
                masked_sq_mean = ((V_td_min_ab ** 2) * nmask).sum() / total
                self._q_t += 1
                beta_t = self._q_beta / (1.0 - (1.0 - self._q_beta) ** self._q_t)
                self._q_mu.data.lerp_(masked_mean,    beta_t)
                self._q_nu.data.lerp_(masked_sq_mean, beta_t)
            q_sigma = (self._q_nu - self._q_mu ** 2).clamp(1e-4, 1e8).sqrt().clamp(1e-4, 1e6)

        # ── V_combined at L_w: slice Q_P from L_full-1 to the window ──
        # Unfold the leading (NAB) axis into separate (NA, NB) axes for clean
        # corner-indexed logging and for the V_combined = (1-α)·V_td + α·Q_I
        # mix at each (α, β) cell.
        alpha_w = rearrange(
            alpha_seq_w[:, :L - 1, :], "(a c b) l 1 -> a c b l 1 1", a=NA, c=NB,
        )                                                                                          # (NA, NB, B, L-1, 1, 1)
        V_td_raw  = rearrange(
            V_td_min_ab, "(a c b) l g 1 -> a c b l g 1", a=NA, c=NB,
        )                                                                                          # (NA, NB, B, L-1, G_sf=1, 1)
        V_td_norm = V_td_raw if not self.normalize_q else (V_td_raw - self._q_mu) / q_sigma

        if has_intrinsic:
            # Q_P (intrinsic critic) carries the full multigamma G_q axis;
            # V_td (SF anchor) is single-gamma.  alpha-mix broadcasts G_sf
            # up to G_q, so V_combined ends up on G_q and gradient through
            # V_td flows only to the deployment actor head (g=-1) via the
            # action slice in maximized_phi_predictor.
            Q_P_norm   = rearrange(
                _gather_ab(Q_P_norm_ab, "L-1"), "(a c b) l g 1 -> a c b l g 1", a=NA, c=NB,
            )                                                                                      # (NA, NB, B, L-1, G_q, 1)
            V_combined = (1.0 - alpha_w) * V_td_norm + alpha_w * Q_P_norm                          # (NA, NB, B, L-1, G_q, 1)
        else:
            # Vanilla SF baseline: no Q_P.  V_combined stays at G_sf=1 and
            # broadcasts to G_q at the masked_avg step (via actor_mask_win).
            V_combined = V_td_norm                                                                  # (NA, NB, B, L-1, G_sf=1, 1)

        online_term = None
        if self.online_coeff > 0:
            online_term = self.online_coeff * -V_combined                                          # has_intrinsic: G_q;  else: G_sf=1

        # a_prime on window (gather from L_full-1) + windowed target predictors.
        a_prime_bklga = rearrange(a_prime_full, "k b l g a -> b l k g a")                        # (B, L_full-1, K_c, G, A_dim)
        a_prime_w     = rearrange(gather_fn(a_prime_bklga, "L-1"), "b l k g a -> k b l g a")     # (K_c, B, L-1, G, A_dim)
        # SF backup runs at G_sf=1 with discount γ_sf (potentially shorter
        # than γ_i to make the SF Bellman objective easier).  Both ``gamma``
        # and ``d`` collapse to a size-1 gamma axis to match the predictor
        # output (single-gamma after the action-axis slice above).
        sf_gamma_t    = self.sf_gammas.to(phi.device).unsqueeze(-1)                              # (G_sf=1, 1)
        d             = repeat(dones.float(), "b l d -> b l 1 1 d")                              # (B, L-1, 1, G_sf=1, 1)
        discount      = sf_gamma_t * (1.0 - d)                                                   # (B, L-1, 1, G_sf=1, 1)

        # SF predictors all run at G_sf=1; slice action tensors to the
        # deployment-head slot ([..., -1:, :]) before each call.  This
        # routes π_{γ_i} actions through the SF Bellman backup at
        # discount γ_sf, making the SF representation model
        # ``V^{π_{γ_i}, γ_sf}`` regardless of how many gamma slots the
        # actor / Q_I are training over.
        with torch.no_grad():
            future_phi = self._ensemble_subset_mean(
                self.target_phi_predictor(
                    next_phi_w, a_prime_w[:, :, :, -1:, :], z_w[:, 1:, ...]
                ).mean(0)
            )                                                                                    # (B, L-1, 1, G_sf=1, psi_dim)
            future_psi = self._ensemble_subset_mean(
                self.target_psi_predictor(
                    next_psi_w, a_prime_w[:, :, :, -1:, :], z_w[:, 1:, ...]
                ).mean(0)
            )                                                                                    # (B, L-1, 1, G_sf=1, phi_dim)

        # Online predictors (where the bulk of the flops live).
        #
        # MSE mode (default): ``forward`` returns the raw network output
        # shaped ``(K=1, B, L, C, G_sf=1, out_dim)``. ``.mean(0)[:, :-1]``
        # collapses the K=1 axis and trims the trailing transition that has
        # no successor (matches the target_*_predictor window).
        #
        # Two-hot mode: ``forward_logits`` returns
        # ``(K=1, B, L, C, G_sf=1, out_dim, n_bins)``; we compute the
        # expected-value tensor (used downstream for V_td/Q_buffer einsums,
        # exactly like ``pred_*`` in MSE mode) from a single network pass,
        # and re-use the cached logits for the cross-entropy in
        # ``regression_loss``.
        if self.phi_predictor.loss_type == "two_hot":
            phi_logits_full = self.phi_predictor.forward_logits(
                phi, a_buffer[:, :, -1:, :].unsqueeze(0), z_w
            )                                                                                    # (1, B, L, C, G_sf=1, psi_dim, n_bins)
            pred_phi = self.phi_predictor._expected_value(phi_logits_full).mean(0)[:, :-1, ...]  # (B, L-1, C, G_sf=1, psi_dim)
            phi_logits = phi_logits_full.mean(0)[:, :-1, ...]                                    # (B, L-1, C, G_sf=1, psi_dim, n_bins)
        else:
            pred_phi = self.phi_predictor(
                phi, a_buffer[:, :, -1:, :].unsqueeze(0), z_w
            ).mean(0)[:, :-1, ...]                                                               # (B, L-1, C, G_sf=1, psi_dim)
            phi_logits = None
        if self.psi_predictor.loss_type == "two_hot":
            psi_logits_full = self.psi_predictor.forward_logits(
                psi, a_buffer[:, :, -1:, :].unsqueeze(0), z_w
            )                                                                                    # (1, B, L, C, G_sf=1, phi_dim, n_bins)
            pred_psi = self.psi_predictor._expected_value(psi_logits_full).mean(0)[:, :-1, ...]  # (B, L-1, C, G_sf=1, phi_dim)
            psi_logits = psi_logits_full.mean(0)[:, :-1, ...]                                    # (B, L-1, C, G_sf=1, phi_dim, n_bins)
        else:
            pred_psi = self.psi_predictor(
                psi, a_buffer[:, :, -1:, :].unsqueeze(0), z_w
            ).mean(0)[:, :-1, ...]                                                               # (B, L-1, C, G_sf=1, phi_dim)
            psi_logits = None

        # next_*_w has no C/G axes; unsqueeze adds them at size 1.  future_*
        # has C=1 (after _ensemble_subset_mean) and G=G_sf=1.  Broadcast
        # gives (B, L-1, C=1, G_sf=1, psi_dim/phi_dim) — the target then
        # broadcasts up to pred_*'s C axis at the regression step.
        target_phi = next_psi_w.unsqueeze(2).unsqueeze(2) + discount * future_phi               # (B, L-1, 1, G_sf=1, psi_dim)
        target_psi = next_phi_w.unsqueeze(2).unsqueeze(2) + discount * future_psi               # (B, L-1, 1, G_sf=1, phi_dim)

        # SF predictors output at G_sf=1; use the collapsed mask to avoid a
        # G_q-broadcast that would silently divide the loss by G_q.
        #
        # MSE mode:
        #   regression_loss(pred=pred_*, target=target_*) is
        #   ``0.5 * (pred - target).pow(2)``.
        # Two-hot mode:
        #   regression_loss(logits=*_logits, target=target_*) returns the
        #   per-coord cross-entropy between log_softmax(logits) and the
        #   two-hot encoding of the (detached) target.
        if self.phi_predictor.loss_type == "two_hot":
            phi_loss_elem = self.phi_predictor.regression_loss(
                logits=phi_logits, target=target_phi.detach(),
            )                                                                                    # (B, L-1, C, G_sf=1, psi_dim)
        else:
            phi_loss_elem = self.phi_predictor.regression_loss(
                pred=pred_phi, target=target_phi.detach(),
            )                                                                                    # (B, L-1, C, G_sf=1, psi_dim)
        if self.psi_predictor.loss_type == "two_hot":
            psi_loss_elem = self.psi_predictor.regression_loss(
                logits=psi_logits, target=target_psi.detach(),
            )                                                                                    # (B, L-1, C, G_sf=1, phi_dim)
        else:
            psi_loss_elem = self.psi_predictor.regression_loss(
                pred=pred_psi, target=target_psi.detach(),
            )                                                                                    # (B, L-1, C, G_sf=1, phi_dim)
        phi_error = utils.masked_avg(phi_loss_elem, sf_critic_mask)
        psi_error = utils.masked_avg(psi_loss_elem, sf_critic_mask)
        critic_loss = phi_error + psi_error

        if self.fb_ortho_coeff > 0:
            psi_ortho_loss = self._ortho_loss("Psi", psi[:, :-1, ...], fb_mask, fb_mask_flat, log_dict=active_log_dict)
            phi_ortho_loss = self._ortho_loss("Phi", phi[:, :-1, ...], fb_mask, fb_mask_flat, log_dict=active_log_dict)
            critic_loss += self._effective_fb_ortho_coeff * (psi_ortho_loss + phi_ortho_loss)

        # Reward regression on full-length psi_full (all-valid BLR regression, not windowed).
        # Convention B: r_t ≈ <ψ(s_{t+1}), z>, so we slice psi_full[:, 1:, :] to match
        # the SF Bellman the predictor was trained against (and the regression in
        # `_compute_reward_z` that produced `z_unnorm_reward`).  `valid_rews` already
        # encodes transition-t validity.
        reward_reg_loss = torch.tensor(0.0, device=phi.device)
        if z_unnorm_reward is not None and valid_rews is not None and valid_rews.any():
            r_target = batch.rews.squeeze(-1) * self.reward_multiplier
            with torch.no_grad():
                pred_r_metric   = (psi_full[:, 1:, :] * z_unnorm_reward.unsqueeze(1)).sum(-1)
                reward_reg_loss = (pred_r_metric[valid_rews] - r_target[valid_rews]).pow(2).mean()
            if self.reward_regression_coeff > 0:
                pred_r        = (psi_full[:, 1:, :] * z_unnorm_reward.detach().unsqueeze(1)).sum(-1)
                ss_res        = (pred_r[valid_rews] - r_target[valid_rews]).pow(2).sum()
                ss_tot        = (r_target[valid_rews] - r_target[valid_rews].mean()).pow(2).sum().clamp(min=1e-8)
                critic_loss   = critic_loss + self.reward_regression_coeff * (ss_res / ss_tot)

        # ── Offline actor term (at L_w; uses normalized per-branch advantages) ──
        offline_term = None
        logp = None
        filter_ = None
        if self.offline_coeff > 0:
            # Q_buffer_td on the window from pred_phi.  pred_phi is single-
            # gamma (G_sf=1), so Q_buffer_td is too; it broadcasts to G_q
            # later via the alpha-mix in A_combined.
            with torch.no_grad():
                Q_buffer_td = torch.einsum(
                    "blcgd,bld->blcg", pred_phi.detach(), z_w[:, :-1, ...]
                ).unsqueeze(-1)                                                                   # (B, L-1, C, G_sf=1, 1)

            # logp: a_dist_multi is a distribution over L_full timesteps (actor ran at
            # L_full for intrinsic-critic targets), so evaluate on full-length buffer
            # actions then gather onto the window.
            a_buffer_full = F.pad(batch.actions.float(), (0, 0, 0, 1), "replicate")               # (B, L_full, A_dim)
            a_buffer_full = repeat(a_buffer_full, f"b l a -> b l {G} a")                          # (B, L_full, G, A_dim)
            a_buffer_full = self.actor.policy_dist.action_from_buffer(a_buffer_full)
            a_buffer_r_full = repeat(a_buffer_full, "b l g a -> (na b) l g a", na=NAB)            # (NAB*B, L_full, G, A_dim)
            if self.discrete:
                logp_ab_full = a_dist_multi.log_prob(a_buffer_r_full.argmax(-1)).unsqueeze(-1)
            elif self.multibinary:
                logp_ab_full = a_dist_multi.log_prob(a_buffer_r_full).mean(-1, keepdim=True)
            else:
                logp_ab_full = a_dist_multi.log_prob(a_buffer_r_full).sum(-1, keepdim=True)      # (NAB*B, L_full, G, 1)
            logp_ab_full_b = rearrange(
                logp_ab_full, "(a b) l g 1 -> b l a g 1", a=NAB,
            )                                                                                      # (B, L_full, NAB, G, 1)
            logp_ab_w      = gather_fn(logp_ab_full_b, "L")                                       # (B, L, NAB, G, 1)
            logp           = rearrange(
                logp_ab_w[:, :-1, ...], "b l (a c) g 1 -> a c b l g 1", a=NA, c=NB,
            )                                                                                      # (NA, NB, B, L-1, G, 1)

            # Per-branch mean-over-critics reduction for the offline advantage,
            # as in AMAGO's MultiTaskAgent.forward: the offline filter uses a
            # mean-ensemble Q while the online DPG term keeps min-over-critics
            # for CDQ pessimism.  Because the mean is linear, alpha-mixing
            # per-branch advantages equals mixing Q/V first and subtracting.
            Q_buf_td_a      = repeat(
                Q_buffer_td, "b l c g 1 -> a c2 b l c g 1", a=NA, c2=NB,
            )                                                                                      # (NA, NB, B, L-1, C, G_sf=1, 1)
            Q_buf_td_mean_a = Q_buf_td_a.mean(4)                                                   # (NA, NB, B, L-1, G_sf=1, 1) raw env-scale

            with torch.no_grad():
                # Per-branch advantages in RAW env-scale (no popart, no EMA).
                # The fbc_filter_func controls scale via its threshold /
                # temperature.  Sign-only filters (binary_filter) are
                # invariant to the popart-affine drop; value-weighted
                # filters get `f(A_raw)` directly on env-scale advantages.
                # Cross-branch scale imbalance only distorts intermediate
                # (α, β) cells; corner cells (pure-α=0, pure-α=1) receive
                # clean single-branch gradients.
                V_td_raw_mean = rearrange(
                    V_td_ab.mean(2), "(a c b) l g 1 -> a c b l g 1", a=NA, c=NB,
                )                                                                                  # (NA, NB, B, L-1, G_sf=1, 1)
                A_TD_raw = Q_buf_td_mean_a - V_td_raw_mean.detach()                                # (NA, NB, B, L-1, G_sf=1, 1)

                if has_intrinsic:
                    Q_buf_ic_w      = rearrange(
                        _gather_ab(Q_buf_ic_ab, "L-1"),
                        "(a c b) l c2 g 1 -> a c b l c2 g 1", a=NA, c=NB,
                    )                                                                              # (NA, NB, B, L-1, C, G_q, 1)
                    Q_buf_ic_mean_w = Q_buf_ic_w.mean(4)                                           # (NA, NB, B, L-1, G_q, 1) raw env-scale
                    Q_P_raw_w       = rearrange(
                        _gather_ab(Q_P_raw_full_ab.mean(2), "L-1"),
                        "(a c b) l g 1 -> a c b l g 1", a=NA, c=NB,
                    )                                                                              # (NA, NB, B, L-1, G_q, 1)
                    A_IC_raw        = Q_buf_ic_mean_w - Q_P_raw_w.detach()                         # (NA, NB, B, L-1, G_q, 1)
                    # Broadcast G_sf=1 (A_TD) up to G_q (A_IC) via alpha-mix.
                    A_combined      = (1.0 - alpha_w) * A_TD_raw + alpha_w * A_IC_raw              # (NA, NB, B, L-1, G_q, 1)
                else:
                    # Vanilla SF baseline: filter is single-gamma; broadcast
                    # to G_q against logp below.
                    A_combined = A_TD_raw                                                          # (NA, NB, B, L-1, G_sf=1, 1)

                filter_ = self.fbc_filter_func(A_combined)                                          # filter shape == A_combined

            # logp always has G_q heads (actor outputs G_q dist params).
            # filter_ broadcasts G_sf=1 -> G_q in the vanilla branch.
            offline_term = self.offline_coeff * -(filter_.detach() * logp)                        # (NA, NB, B, L-1, G_q, 1)

        # ═══════════════════════════════════════════════════════════════════
        # Assemble final losses + logging
        # ═══════════════════════════════════════════════════════════════════
        # Both online and offline actor terms live at L_w-1; use the same mask.
        # actor_mask_win carries G_q; broadcast handles the G_sf=1 vanilla
        # SF case (V_combined collapses to G_sf=1 when has_intrinsic=False).
        actor_mask_win = actor_mask_w[:, :-1, ...]    # (B, L-1, G_q, 1)

        actor_loss_scalar = torch.zeros((), device=phi_full.device)
        if online_term is not None:
            actor_loss_scalar = actor_loss_scalar + utils.masked_avg(
                online_term.mean(dim=(0, 1)), actor_mask_win,
            )
        if offline_term is not None:
            actor_loss_scalar = actor_loss_scalar + utils.masked_avg(
                offline_term.mean(dim=(0, 1)), actor_mask_win,
            )

        if log_step:
            z_diag = self._z_diagnostics(
                z_reward_norm, z_random, z_goal, z_unnorm_reward,
                psi_full, valid_rews, batch.rews,
            )
            self.update_info.update({
                "Jepa/Loss": critic_loss,
                "Jepa/Psi/Error": psi_error,
                "Jepa/Phi/Error": phi_error,
                "Reward/Regression MSE": reward_reg_loss.item(),
                "SeqLen/Backbone":        L_full,  # dynamics backbone (§ 1)
                "SeqLen/TrajEncoderPi":   L_full,  # policy traj encoder + actor forward (§ 2)
                "SeqLen/IntrinsicCritic": L_full,  # intrinsic stack (§ 2)
                "SeqLen/PsiPredictor":    L,       # phi/psi/target_* predictors (windowed § 3)
                "SeqLen/ActorLoss":       L,       # V_td / V_combined / offline_term (windowed § 3)
                "Batch Size (in Timesteps)": B * L,
                "Unmasked Batch Size (in Timesteps)": actor_mask_win[..., 0, 0].sum(),
                "FB Mask Valid": fb_mask_flat.sum(),
            })
            self.update_info.update(z_diag)

            # Per-coord encoder + TD-target summary stats. ``L2NormMean`` is
            # the sphere-norm sanity check: should track ≈ ``sqrt(emb_dim)``
            # for fb-normed encoders.  ``AbsMax`` on the targets drives the
            # two-hot bin range choice — pair with ``TwoHotPhi/EdgeFraction``
            # (when two-hot) to decide if the range needs widening.
            with torch.no_grad():
                def _stats(name: str, x: torch.Tensor) -> dict:
                    return {
                        f"{name}/Mean":       x.mean().item(),
                        f"{name}/Min":        x.min().item(),
                        f"{name}/Max":        x.max().item(),
                        f"{name}/AbsMax":     x.abs().max().item(),
                        f"{name}/L2NormMean": x.norm(dim=-1).mean().item(),
                    }
                self.update_info.update(_stats("Phi",       phi_full))
                self.update_info.update(_stats("Psi",       psi_full))
                self.update_info.update(_stats("TargetPhi", target_phi))
                self.update_info.update(_stats("TargetPsi", target_psi))
                # Two-hot predicted-bin histogram (argmax of the per-coord
                # logits), mirroring AMAGO's ``MultiTaskAgent._td_stats``
                # and the ``IntrinsicQ/Maximum Bin (...)`` logging in
                # ``IntrinsicCriticStack.forward``.  Targets aren't logged
                # separately; ``TwoHot*/EdgeFraction`` covers target-side
                # clamping.
                #
                # ``phi_logits`` / ``psi_logits`` are only populated in
                # two-hot mode (shape ``(B, L-1, C, G_sf=1, out_dim,
                # n_bins)``); ``sf_critic_mask`` broadcasts across
                # ``out_dim`` so every valid per-coord prediction
                # contributes one histogram sample.
                def _two_hot_pred_bin_histogram(
                    name: str, logits: torch.Tensor, n_bins: int
                ) -> None:
                    try:
                        import wandb as _wandb
                        pred_bin = logits.argmax(-1)                            # (B, L-1, C, G, out_dim)
                        mask = sf_critic_mask.expand_as(pred_bin).bool()
                        pred_flat = pred_bin[mask]
                        if pred_flat.numel() > 0:
                            self.update_info[f"TwoHot{name}/PredBin"] = (
                                _wandb.Histogram(
                                    pred_flat.detach().cpu().numpy(),
                                    num_bins=n_bins,
                                )
                            )
                    except Exception:
                        # Histograms are purely diagnostic; never let
                        # a wandb import / serialization issue kill
                        # the train step.
                        pass

                if self.phi_predictor.loss_type == "two_hot":
                    bv = self.phi_predictor.bin_vals
                    self.update_info["TwoHotPhi/EdgeFraction"] = (
                        ((target_phi <= bv[0]) | (target_phi >= bv[-1]))
                        .float().mean().item()
                    )
                    _two_hot_pred_bin_histogram(
                        "Phi", phi_logits, self.phi_predictor.num_bins,
                    )
                if self.psi_predictor.loss_type == "two_hot":
                    bv = self.psi_predictor.bin_vals
                    self.update_info["TwoHotPsi/EdgeFraction"] = (
                        ((target_psi <= bv[0]) | (target_psi >= bv[-1]))
                        .float().mean().item()
                    )
                    _two_hot_pred_bin_histogram(
                        "Psi", psi_logits, self.psi_predictor.num_bins,
                    )

            # Per-corner online-term / V_td / Q_P diagnostics (at L_w-1).
            # Always log the (α=0, β=0) corner (slot (0, 0)).  When NA ≥ 2
            # also log (α_max, β=0); when both NA ≥ 2 and NB ≥ 2 also log
            # (α_max, β_max).  These three pinned corners cover the
            # three meaningful regimes:
            #   (0, 0)         → pure SF (V_combined ≡ V_td)
            #   (α_max, 0)     → pure intrinsic curiosity (Q_I trained on r_int)
            #   (α_max, β_max) → pure extrinsic (Q_I trained on r_ext)
            with torch.no_grad():
                corner_slices = [("alpha0", 0, 0)]
                if NA >= 2:
                    corner_slices.append(("alpha_max", -1, 0))
                if NA >= 2 and NB >= 2:
                    corner_slices.append(("alpha_max_beta_max", -1, -1))
                # Split per-gamma logging into the deployment head (g=-1,
                # the γ_i actor slot used at rollout time + as the SF
                # backup policy) and the auxiliary multigamma heads
                # (g=0..G_q-2, regularizers via parameter sharing).
                # V_td has G_sf=1 so it has no auxiliary split — log
                # once.  Q_P / V_combined are per-gamma over G_q.
                G_q       = len(self.gammas)
                has_aux_g = G_q > 1
                actor_mask_dep = actor_mask_win[..., -1:, :]                     # (B, L-1, 1, 1)
                if has_aux_g:
                    actor_mask_aux = actor_mask_win[..., :-1, :]                  # (B, L-1, G_q-1, 1)
                for name, ai, ci in corner_slices:
                    if online_term is not None:
                        self.update_info[f"ActorLoss/{name}"] = utils.masked_avg(
                            online_term[ai, ci], actor_mask_win,
                        )
                    self.update_info[f"ActorLoss/V_td_{name}"] = utils.masked_avg(
                        V_td_norm[ai, ci], actor_mask_win,
                    )
                    if has_intrinsic:
                        # Deployment-head Q_P / V_combined (the only series
                        # that maps cleanly onto a stationary value
                        # function for π_{γ_i}; safe to compare across
                        # baselines).
                        self.update_info[f"ActorLoss/Q_P_{name}"] = utils.masked_avg(
                            Q_P_norm[ai, ci, ..., -1:, :], actor_mask_dep,
                        )
                        self.update_info[f"ActorLoss/V_combined_{name}"] = utils.masked_avg(
                            V_combined[ai, ci, ..., -1:, :], actor_mask_dep,
                        )
                        if has_aux_g:
                            # Auxiliary heads optimize different
                            # non-stationary objectives; logged
                            # separately to avoid mixing scales in the
                            # primary V_combined dashboard.
                            self.update_info[f"ActorLoss/Q_P_aux_{name}"] = utils.masked_avg(
                                Q_P_norm[ai, ci, ..., :-1, :], actor_mask_aux,
                            )
                            self.update_info[f"ActorLoss/V_combined_aux_{name}"] = utils.masked_avg(
                                V_combined[ai, ci, ..., :-1, :], actor_mask_aux,
                            )

            self.update_info.update(self._policy_stats(actor_mask_full, a_dist))

            if online_term is not None and offline_term is not None:
                on_mag  = utils.masked_avg(online_term .mean(dim=(0, 1)).abs(), actor_mask_win)
                off_mag = utils.masked_avg(offline_term.mean(dim=(0, 1)).abs(), actor_mask_win)
                self.update_info.update({
                    "ActorLoss/online_term_mag":   on_mag,
                    "ActorLoss/offline_term_mag":  off_mag,
                    "ActorLoss/online_offline_ratio": on_mag / off_mag.clamp(min=1e-8),
                })
            elif online_term is not None:
                self.update_info["ActorLoss/online_term_mag"] = utils.masked_avg(
                    online_term.mean(dim=(0, 1)).abs(), actor_mask_win,
                )
            elif offline_term is not None:
                self.update_info["ActorLoss/offline_term_mag"] = utils.masked_avg(
                    offline_term.mean(dim=(0, 1)).abs(), actor_mask_win,
                )

            if self.offline_coeff > 0 and logp is not None and filter_ is not None:
                # logp / filter_ are (NA, NB, B, L-1, G, 1) — log the (0, 0)
                # corner stats, plus an "AlphaMax/" series for the (α_max,
                # β=0) corner when NA ≥ 2.
                self.update_info.update(self._filter_stats(actor_mask_w, logp[0, 0], filter_[0, 0]))
                if NA >= 2:
                    stats_max = self._filter_stats(actor_mask_w, logp[-1, 0], filter_[-1, 0])
                    self.update_info.update({f"AlphaMax/{k}": v for k, v in stats_max.items()})

            if self.normalize_q:
                q_sigma_log = (self._q_nu - self._q_mu ** 2).clamp(1e-4, 1e8).sqrt().clamp(1e-4, 1e6)
                self.update_info.update({
                    "NormQ/mu":                self._q_mu.item(),
                    "NormQ/sigma":             q_sigma_log.item(),
                    "NormQ/V_agent_raw_mean":  V_td_min_ab.mean().item(),
                })
            if has_intrinsic:
                # alphas / betas are (NA, B, 1) and (NB, B, 1) with per-B
                # independent draws; log the per-slot mean across B.  Slot 0
                # / slot -1 are pinned (when the respective axis has ≥ 2
                # slots), middle slots vary across B.  V_combined is
                # averaged over BOTH (NA, NB) axes.
                self.update_info.update({
                    "IntrinsicQ/Alpha Values":     alphas.mean(dim=1).squeeze(-1).tolist(),
                    "IntrinsicQ/Beta Values":      betas .mean(dim=1).squeeze(-1).tolist(),
                    "IntrinsicQ/Alpha Mean":       alphas.mean().item(),
                    "IntrinsicQ/Beta Mean":        betas .mean().item(),
                    "IntrinsicQ/V_combined Mean":  utils.masked_avg(
                        V_combined.mean(dim=(0, 1)), actor_mask_win,
                    ),
                })
                try:
                    import wandb
                    self.update_info["IntrinsicQ/Alpha Histogram"] = wandb.Histogram(
                        alphas.detach().cpu().float().flatten().numpy(), num_bins=32
                    )
                    self.update_info["IntrinsicQ/Beta Histogram"] = wandb.Histogram(
                        betas.detach().cpu().float().flatten().numpy(), num_bins=32
                    )
                except Exception:
                    pass
                self.update_info.update(self.intrinsic_stack.update_info)

        intrinsic_critic_loss_out = None
        if has_intrinsic:
            intrinsic_critic_loss_out = intrinsic_critic_loss

        return critic_loss, actor_loss_scalar, intrinsic_critic_loss_out

    def init_hidden_state(self, batch_size: int, device: torch.device):
        """Extend base (phi, psi) tuple with slots for traj_encoder_pi and the
        β-gated extrinsic_traj_encoder.  The 4th element is ``None`` when the
        extrinsic branch is auto-skipped (β≡0 baselines)."""
        h_ext = (
            self.extrinsic_traj_encoder.init_hidden_state(batch_size, device)
            if self.extrinsic_traj_encoder is not None
            else None
        )
        return (
            self.traj_encoder.init_hidden_state(batch_size, device),
            self.traj_encoder_b.init_hidden_state(batch_size, device),
            self.traj_encoder_pi.init_hidden_state(batch_size, device),
            h_ext,
        )

    def reset_hidden_state(self, hidden_state, dones: np.ndarray):
        if hidden_state is None:
            return None
        h_a, h_b, h_pi, h_ext = hidden_state
        h_ext_new = (
            self.extrinsic_traj_encoder.reset_hidden_state(h_ext, dones)
            if self.extrinsic_traj_encoder is not None and h_ext is not None
            else None
        )
        return (
            self.traj_encoder.reset_hidden_state(h_a, dones),
            self.traj_encoder_b.reset_hidden_state(h_b, dones),
            self.traj_encoder_pi.reset_hidden_state(h_pi, dones),
            h_ext_new,
        )

    def get_actions(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state=None,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, Any]:
        hidden_state_phi, hidden_state_psi, hidden_state_pi, hidden_state_ext = (
            hidden_state
        )
        B_inf, L_inf = next(iter(obs.values())).shape[:2]
        device = next(iter(obs.values())).device
        if self._inference_alpha is not None:
            alpha_inf = self._inference_alpha.unsqueeze(1).expand(-1, L_inf, -1)
        elif self.intrinsic_stack is not None:
            raise RuntimeError(
                "_inference_alpha is None but intrinsic stack is enabled. "
                "Call propose_alpha_for_collection() or set _inference_alpha before get_actions()."
            )
        else:
            alpha_inf = torch.zeros(B_inf, L_inf, 1, device=device)
        # β rollout: defaults to 0 when unset (consistent with the off-policy
        # collection convention for non-β baselines).  When the extrinsic
        # branch is constructed (β_dist.max_value > 0), callers MUST set
        # ``_inference_beta`` via propose_beta_for_collection or the eval
        # phase plumbing.
        if self._inference_beta is not None:
            beta_inf = self._inference_beta.unsqueeze(1).expand(-1, L_inf, -1)
        elif self.extrinsic_traj_encoder is not None:
            raise RuntimeError(
                "_inference_beta is None but the extrinsic encoder is "
                "constructed.  Call propose_beta_for_collection() or set "
                "_inference_beta before get_actions()."
            )
        else:
            beta_inf = torch.zeros(B_inf, L_inf, 1, device=device)
        # phi backbone (primary) -- drives policy
        tstep_emb = self.tstep_encoder(obs=obs, rl2s=rl2s)
        traj_emb_t, hidden_state_phi = self.traj_encoder(
            tstep_emb, time_idxs=time_idxs, hidden_state=hidden_state_phi
        )
        # psi backbone (secondary) -- needed for online BLR z updates
        tstep_emb_b = self.tstep_encoder_b(obs=obs, rl2s=rl2s)
        traj_emb_b, hidden_state_psi = self.traj_encoder_b(
            tstep_emb_b, time_idxs=time_idxs, hidden_state=hidden_state_psi
        )
        self._last_traj_emb_b = traj_emb_b

        B, L = traj_emb_t.shape[:2]
        if self.inference_z is None:
            raise RuntimeError("inference_z is None. Call InferenceMode.init_z first.")

        if self.inference_z.ndim == 1:
            print("using single z")
            z = self.inference_z.unsqueeze(0).unsqueeze(0).expand(B, L, -1)
        else:
            num_z = self.inference_z.shape[0]
            if B != num_z:
                raise RuntimeError(f"Batch size ({B}) != bootstrap samples ({num_z})")
            z = self.inference_z.unsqueeze(1).expand(-1, L, -1)
        phi = self.phi_encoder(traj_emb_t)
        # tstep_encoder_pi projects (phi, intrinsic_reward, prev_action) per
        # step; traj_encoder_pi then attends over the resulting features.
        # ``obs`` already contains ``intrinsic_reward`` (populated by
        # ``BilevelAgent.get_actions`` to match the training-time key) and
        # ``prev_action`` (standard AMAGO env-wrapper key).
        pi_input = self.tstep_encoder_pi(phi=phi, obs=obs)  # (B, L, pi_tstep_dim)
        policy_ctx, hidden_state_pi = self.traj_encoder_pi(
            pi_input, time_idxs=time_idxs, hidden_state=hidden_state_pi
        )

        # Extrinsic-reward branch (β-gated).  Skipped entirely when the ext
        # encoder isn't constructed (β≡0 baselines), in which case
        # ``gated_ext`` has 0 trailing dims and contributes nothing to the
        # actor input.  When constructed, the extrinsic encoder consumes
        # obs["observed_normalized"], obs["prev_action"] and obs["ext_reward"]
        # and projects to a small per-step embedding; β-gating zeros the
        # contribution at β=0 cells regardless of encoder content.
        if self.extrinsic_traj_encoder is not None:
            ext_tstep = self.extrinsic_tstep_encoder(obs=obs, rl2s=rl2s)
            ext_ctx, hidden_state_ext = self.extrinsic_traj_encoder(
                ext_tstep,
                time_idxs=time_idxs,
                hidden_state=hidden_state_ext,
            )
            gated_ext = beta_inf * ext_ctx
        else:
            gated_ext = torch.zeros(B, L, 0, device=device)

        new_hidden = (
            hidden_state_phi,
            hidden_state_psi,
            hidden_state_pi,
            hidden_state_ext,
        )

        # Fourier-encode α and β scalars before they enter the actor MLP.
        # See metabfm.rl.conditioning for the encoding rationale.
        alpha_inf_enc = fourier_encode(alpha_inf, self.fourier_n_freq)  # (B, L, 2*nf+1)
        beta_inf_enc = fourier_encode(beta_inf, self.fourier_n_freq)

        action_dists = self.actor(
            self._cat_actor_input(
                policy_ctx,
                z,
                alpha_inf_enc,
                beta_inf_enc,
                gated_ext,
            ),
            straight_from_obs={k: obs[k] for k in self.pass_obs_keys_to_actor},
        )
        if sample:
            actions = action_dists.sample()
        else:
            if self.discrete:
                actions = torch.argmax(action_dists.probs, dim=-1, keepdim=True)
            else:
                actions = action_dists.mean
        actions = actions[..., -1, :]
        dtype = torch.uint8 if (self.discrete or self.multibinary) else torch.float32
        return actions.to(dtype=dtype), new_hidden

    @torch.no_grad()
    def extract_phi_for_blr(
        self, actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Psi features from cached traj_emb_b for BLR update. Returns (B, psi_dim)."""
        return self.psi_encoder(self._last_traj_emb_b[:, -1:, :]).squeeze(1)

    def normalize_z(self, z_unnorm: torch.Tensor) -> torch.Tensor:
        return self.psi_encoder.out_norm(z_unnorm)

    def _sample_z(self, batch_size: int, device: torch.device) -> torch.Tensor:
        random_z = torch.randn(
            batch_size, self.psi_dim, dtype=torch.float32, device=device
        )
        with torch.no_grad():
            z = self.psi_encoder.out_norm(random_z)
        return z

    @torch.no_grad()
    def _compute_reward_z(
        self,
        psi_full: torch.Tensor,
        rews_full: torch.Tensor,
        valid_rews: torch.Tensor,
        use_reward: torch.Tensor,
        B: int,
        L_full: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute z via BLR Thompson Sampling on full-sequence rewards.

        Returns (z_norm, z_unnorm) each (B, psi_dim). z_unnorm is always from
        the full-sequence BLR solve (used as regression target). z_norm may come
        from a truncated-prefix BLR if z_reward_intermediate is True.
        """
        from metabfm.rl.blr import BLR

        infer = self.inference_mode
        assert isinstance(infer, OnlineBLR)

        # Convention B (matches our SF Bellman target ψ(s_{t+1}) + γ M̄(s', a', z)):
        # the trained predictor satisfies <ψ(s_{t+1}), z> ≈ r_t, so z is regressed
        # against ψ at the *next* state, not the originating state.  The validity
        # mask `valid_rews` already represents transition-t validity (= s_{t+1}
        # non-padded AND r_t != NO_REWARD), so it indexes correctly under [:, 1:, :].
        psi_obs = psi_full[:, 1:, :]
        r_scaled = rews_full.squeeze(-1) * self.reward_multiplier
        full_mask = valid_rews.float()

        blr = BLR.create_LSQ(
            dim=self.psi_dim, lam=infer.lam, batch_size=B, device=psi_full.device
        )
        blr.batch_update(
            psi_obs * full_mask.unsqueeze(-1), r_scaled * full_mask, sigma=infer.sigma
        )
        z_unnorm = blr.sample()

        if self.z_reward_intermediate:
            valid_counts = valid_rews.float().sum(dim=1).clamp(min=1)
            rand_k = (torch.rand(B, device=psi_full.device) * valid_counts).long()
            cumvalid = valid_rews.float().cumsum(dim=1)
            target = (rand_k + 1).float().unsqueeze(1)
            chosen_t = (cumvalid >= target).float().argmax(dim=1)
            time_idx = torch.arange(L_full - 1, device=psi_full.device).unsqueeze(0)
            trunc_mask = (time_idx <= chosen_t.unsqueeze(1)).float()
            trunc_mask = full_mask * trunc_mask
            blr_trunc = BLR.create_LSQ(
                dim=self.psi_dim, lam=infer.lam, batch_size=B, device=psi_full.device
            )
            blr_trunc.batch_update(
                psi_obs * trunc_mask.unsqueeze(-1),
                r_scaled * trunc_mask,
                sigma=infer.sigma,
            )
            z_norm = self.normalize_z(blr_trunc.sample())
        else:
            z_norm = self.normalize_z(z_unnorm)

        return z_norm, z_unnorm

    @torch.no_grad()
    def _z_diagnostics(
        self,
        z_reward_norm: Optional[torch.Tensor],
        z_random: torch.Tensor,
        z_goal: Optional[torch.Tensor],
        z_unnorm_reward: Optional[torch.Tensor],
        psi_full: torch.Tensor,
        valid_rews: Optional[torch.Tensor],
        rews: torch.Tensor,
    ) -> dict:
        """Diagnostic metrics comparing reward, goal, and random z distributions."""
        diag = {}

        def _r_squared(
            z: torch.Tensor,
            psi_obs: torch.Tensor,
            r_target: torch.Tensor,
            mask: torch.Tensor,
        ) -> float:
            pred = (psi_obs * z.unsqueeze(1)).sum(-1)
            pred_valid = pred[mask]
            r_valid = r_target[mask]
            if r_valid.numel() < 2:
                return 0.0
            ss_res = (pred_valid - r_valid).pow(2).sum()
            ss_tot = (r_valid - r_valid.mean()).pow(2).sum()
            if ss_tot < 1e-8:
                return 0.0
            return (1.0 - ss_res / ss_tot).clamp(-1.0, 1.0).item()

        diag["Z/Random Pairwise Cosine"] = pairwise_cosine(z_random)
        if z_reward_norm is not None:
            diag["Z/Reward Pairwise Cosine"] = pairwise_cosine(z_reward_norm)
        if z_goal is not None:
            diag["Z/Goal Pairwise Cosine"] = pairwise_cosine(z_goal)

        if valid_rews is not None and valid_rews.any() and z_unnorm_reward is not None:
            # Convention B: r_t ≈ <ψ(s_{t+1}), z>, matching `_compute_reward_z`
            # and the trained Bellman target.
            psi_obs = psi_full[:, 1:, :].detach()
            r_target = rews.squeeze(-1) * self.reward_multiplier
            diag["Z/Reward R²"] = _r_squared(
                z_unnorm_reward, psi_obs, r_target, valid_rews
            )
            diag["Z/Random R²"] = _r_squared(z_random, psi_obs, r_target, valid_rews)
            if z_goal is not None:
                diag["Z/Goal R²"] = _r_squared(z_goal, psi_obs, r_target, valid_rews)

        return diag

    @property
    def task_vector_dim(self) -> int:
        return self.psi_dim

    @property
    def trainable_params(self):
        params = [
            self.tstep_encoder.parameters(),
            self.traj_encoder.parameters(),
            self.tstep_encoder_b.parameters(),
            self.traj_encoder_b.parameters(),
            self.tstep_encoder_pi.parameters(),
            self.traj_encoder_pi.parameters(),
            self.phi_encoder.parameters(),
            self.psi_encoder.parameters(),
            self.phi_predictor.parameters(),
            self.psi_predictor.parameters(),
            self.actor.parameters(),
        ]
        if self.extrinsic_tstep_encoder is not None:
            params.append(self.extrinsic_tstep_encoder.parameters())
        if self.extrinsic_traj_encoder is not None:
            params.append(self.extrinsic_traj_encoder.parameters())
        if self.intrinsic_stack is not None:
            params.append(self.intrinsic_stack.parameters())
        return itertools.chain(*params)

    def clip_param_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """Partition trainable params into AMAGO / TDJepa-core / RND groups.

        Disjoint by construction (gradient subgraphs are already separated by
        ``detach_actor=True`` cutting phi at the actor input, sanitize_published
        cutting dynamics outputs, the SF loss using buffer actions, and RND's
        target/input detaches). Used by ``BilevelExperiment.train_step`` to
        apply distinct clip primitives per subsystem:

          * ``amago``       — actor + IC critics + popart + policy-side encoders
                              + extrinsic encoders. Treated like a self-contained
                              AMAGO instance and norm-clipped.
          * ``tdjepa_core`` — phi/psi backbones + heads + JEPA predictors.
                              Per-element value-clipped as a stability
                              backstop.
          * ``rnd``         — RND predictor only. Norm-clipped.
        """
        amago: list[torch.nn.Parameter] = list(
            itertools.chain(
                self.tstep_encoder_pi.parameters(),
                self.traj_encoder_pi.parameters(),
                self.actor.parameters(),
            )
        )
        if self.extrinsic_tstep_encoder is not None:
            amago.extend(self.extrinsic_tstep_encoder.parameters())
        if self.extrinsic_traj_encoder is not None:
            amago.extend(self.extrinsic_traj_encoder.parameters())
        rnd: list[torch.nn.Parameter] = []
        if self.intrinsic_stack is not None:
            ic_groups = self.intrinsic_stack.clip_param_groups()
            amago.extend(ic_groups["amago"])
            rnd.extend(ic_groups["rnd"])
        tdjepa_core: list[torch.nn.Parameter] = list(
            itertools.chain(
                self.tstep_encoder.parameters(),
                self.traj_encoder.parameters(),
                self.tstep_encoder_b.parameters(),
                self.traj_encoder_b.parameters(),
                self.phi_encoder.parameters(),
                self.psi_encoder.parameters(),
                self.phi_predictor.parameters(),
                self.psi_predictor.parameters(),
            )
        )
        return {"amago": amago, "tdjepa_core": tdjepa_core, "rnd": rnd}

    def hard_sync_targets(self):
        self._full_copy(self.target_phi_encoder, self.phi_encoder)
        self._full_copy(self.target_psi_encoder, self.psi_encoder)
        self._full_copy(self.target_phi_predictor, self.phi_predictor)
        self._full_copy(self.target_psi_predictor, self.psi_predictor)
        if self.intrinsic_stack is not None:
            self.intrinsic_stack.hard_sync_targets()

    def soft_sync_targets(self):
        self._ema_copy(self.target_phi_encoder, self.phi_encoder)
        self._ema_copy(self.target_psi_encoder, self.psi_encoder)
        self._ema_copy(self.target_phi_predictor, self.phi_predictor)
        self._ema_copy(self.target_psi_predictor, self.psi_predictor)
        self._full_copy(self.maximized_phi_predictor, self.phi_predictor)
        if self.intrinsic_stack is not None:
            self.intrinsic_stack.soft_sync_targets()

    def get_grad_norms(self) -> dict[str, float]:
        norms = {
            "Actor Grad Norm": utils.get_grad_norm(self.actor),
            "TstepEncoder (phi) Grad Norm": utils.get_grad_norm(self.tstep_encoder),
            "TrajEncoder (phi) Grad Norm": utils.get_grad_norm(self.traj_encoder),
            "TstepEncoder (psi) Grad Norm": utils.get_grad_norm(self.tstep_encoder_b),
            "TrajEncoder (psi) Grad Norm": utils.get_grad_norm(self.traj_encoder_b),
            "TstepEncoder (pi) Grad Norm": utils.get_grad_norm(self.tstep_encoder_pi),
            "TrajEncoder (pi) Grad Norm": utils.get_grad_norm(self.traj_encoder_pi),
            "Phi Encoder Grad Norm": utils.get_grad_norm(self.phi_encoder),
            "Psi Encoder Grad Norm": utils.get_grad_norm(self.psi_encoder),
            "Phi Predictor Grad Norm": utils.get_grad_norm(self.phi_predictor),
            "Psi Predictor Grad Norm": utils.get_grad_norm(self.psi_predictor),
        }
        if self.intrinsic_stack is not None:
            s = self.intrinsic_stack
            norms["IntrinsicQ/Grad Norm"] = utils.get_grad_norm(s)
            norms["IntrinsicQ/Critics Grad Norm"] = utils.get_grad_norm(s.critics)
            norms["IntrinsicQ/RND Predictor Grad Norm"] = utils.get_grad_norm(
                s.rnd_predictor
            )
        return norms
