"""ExtrinsicTstepEncoder: per-timestep encoder for the β-gated extrinsic-reward
context branch of the hybrid α-β ZSRL agent.

Lives in its own module so gin registration is canonical under
``metabfm.rl.extrinsic_encoder.*`` regardless of how the launcher is
invoked.  Its output is sequence-modeled by a separate
``extrinsic_traj_encoder`` (any AMAGO ``TrajEncoder``, set via gin scope).

Normalization: ``observed_normalized`` (z-scored by :class:`BilevelAgent`)
and ``prev_action`` (``[-1, 1]`` or one-hot) pass through; the raw
``ext_reward`` goes through an :class:`InputNorm` whose stats update only
in training mode (frozen at rollout / eval).

``hide_rl2s=True`` is mandatory, as for every TDJepa-side encoder:
``ext_reward`` is read only from ``obs["ext_reward"]``, which
``BilevelAgent`` copies out of rl2s.
"""

from __future__ import annotations

import math
from typing import Optional

import gin
import gym
import torch
from torch import nn

from amago.nets import ff
from amago.nets.tstep_encoders import TstepEncoder
from amago.nets.utils import InputNorm


@gin.configurable
class ExtrinsicTstepEncoder(TstepEncoder):
    """β-gated extrinsic-reward branch tstep encoder.

    Projects ``cat([observed_normalized, prev_action, InputNorm(ext_reward)])``
    through a small MLP to ``tstep_dim``.

    Keyword Args:
        tstep_dim: Output embedding dimension (intentionally smaller than
            the 256-d policy tstep encoder).
        n_layers: MLP depth; 1 keeps per-step compute cheap.
        d_hidden: MLP hidden dimension.
        activation: Activation between MLP layers.
        normalize_ext_reward: Apply :class:`InputNorm` to ``ext_reward``.
            Disable only for ablations.
    """

    REQUIRED_OBS_KEYS = ("observed_normalized", "prev_action", "ext_reward")

    def __init__(
        self,
        obs_space: gym.Space,
        rl2_space: gym.Space,
        tstep_dim: int = 128,
        n_layers: int = 1,
        d_hidden: int = 128,
        activation: str = "leaky_relu",
        normalize_ext_reward: bool = True,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space, hide_rl2s=True)

        for k in self.REQUIRED_OBS_KEYS:
            if k not in obs_space.spaces:
                raise ValueError(
                    f"ExtrinsicTstepEncoder requires obs key '{k}' but it is "
                    f"missing from obs_space.spaces (have: "
                    f"{sorted(obs_space.spaces.keys())}).  Check that "
                    f"BilevelAgent registers all three keys."
                )

        obs_dim = int(math.prod(obs_space.spaces["observed_normalized"].shape))
        prev_action_dim = int(math.prod(obs_space.spaces["prev_action"].shape))
        ext_reward_dim = int(math.prod(obs_space.spaces["ext_reward"].shape))
        assert ext_reward_dim == 1, (
            f"ExtrinsicTstepEncoder expects scalar ext_reward (dim=1), "
            f"got dim={ext_reward_dim}."
        )

        self.ext_reward_norm = InputNorm(ext_reward_dim, skip=not normalize_ext_reward)

        in_dim = obs_dim + prev_action_dim + ext_reward_dim
        self._in_dim = in_dim

        self.base = ff.MLP(
            d_inp=in_dim,
            d_hidden=d_hidden,
            n_layers=n_layers,
            d_output=tstep_dim,
            activation=activation,
            normalization=None,
        )
        self.out_norm = ff.Normalization("layer", tstep_dim)
        self._emb_dim = tstep_dim

    def inner_forward(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        log_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        # rl2s is zeroed by the base ``forward`` (hide_rl2s); ext_reward comes
        # from obs["ext_reward"], which BilevelAgent populates from
        # rl2s[..., 0:1] in both train and rollout paths.
        assert (
            self.hide_rl2s
        ), "ExtrinsicTstepEncoder requires hide_rl2s=True; got False."
        del rl2s, log_dict

        observed = obs["observed_normalized"].float()
        prev_action = obs["prev_action"].float()
        ext_reward = obs["ext_reward"].float()
        if observed.ndim == 2:
            observed = observed.unsqueeze(-1)
        observed = observed.flatten(start_dim=2)
        if prev_action.ndim == 2:
            prev_action = prev_action.unsqueeze(-1)
        prev_action = prev_action.flatten(start_dim=2)

        if self.training:
            self.ext_reward_norm.update_stats(ext_reward)
        ext_reward_norm = self.ext_reward_norm(ext_reward)

        x = torch.cat([observed, prev_action, ext_reward_norm], dim=-1)
        return self.out_norm(self.base(x))

    @property
    def emb_dim(self) -> int:
        return self._emb_dim

    @property
    def in_dim(self) -> int:
        return self._in_dim
