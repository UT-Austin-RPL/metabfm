"""PolicyTstepEncoder: per-timestep encoder shared by the actor and the
intrinsic critic.

Lives in its own module so gin registration is canonical under
``metabfm.rl.policy_tstep.*`` regardless of how the launcher is invoked.
"""

from __future__ import annotations

import gin
import torch
from torch import nn

from amago.nets import ff


_POLICY_TSTEP_ENCODER_ALLOWED_KEYS = frozenset(
    {
        "intrinsic_reward",
        "prev_action",
        "rldp",
        "idm",
        "observed_normalized",
        "rldp_error",
    }
)


@gin.configurable
class PolicyTstepEncoder(nn.Module):
    """Per-timestep encoder shared by the actor and the intrinsic critic.

    A learned projection of ``cat([phi, *extras])``, giving the actor a
    gradient-accessible path to the extras it conditions on (e.g.
    ``prev_action``) even when phi is detached (``detach_actor=True``).  The
    intrinsic critic has no tstep/traj encoder of its own; it reuses the
    actor's ``policy_ctx`` (the output of ``traj_encoder_pi``), so this is
    the only learned per-step projection on the actor + intrinsic-critic side.

    Inputs
    ------
    phi : (B, L, phi_dim)
        Always included.  Passed as ``detach_if(phi)`` in the training path;
        the encoder's own parameters still receive actor + intrinsic-critic
        gradients.
    obs : dict[str, torch.Tensor]
        Must contain every key in ``specify_obs_keys`` (a subset of
        ``_CONCAT_ORDER``).  Concat order follows ``_CONCAT_ORDER``,
        regardless of the gin list order.

    No ``InputNorm`` is applied: every input is bounded or normalized at
    source.  ``phi`` by the phi-encoder ``out_norm``; ``intrinsic_reward`` is
    RMS-normalized by ``IntrinsicCriticStack``; ``prev_action`` is in
    ``[-1, 1]`` / one-hot; ``rldp`` and ``idm`` are sphere-normalized to
    radius ``sqrt(d)`` by ``DynamicsEncoder`` (enabled in
    ``dynamics/latent_idm.gin``); ``observed_normalized`` is z-scored by
    ``BilevelAgent``; ``rldp_error`` is a cosine distance in ``[0, 2]``.

    ``auto_scale`` (default ``True``) mirrors AMAGO's ``FFTstepEncoder``:
    ``d_hidden`` and ``d_output`` are clamped up to at least ``in_dim`` so
    the MLP never bottlenecks a wide input.
    """

    _CONCAT_ORDER = (
        "intrinsic_reward",
        "prev_action",
        "rldp",
        "idm",
        "observed_normalized",
        "rldp_error",
    )

    def __init__(
        self,
        phi_dim: int,
        prev_action_dim: int,
        rldp_dim: int = 0,
        idm_dim: int = 0,
        observed_normalized_dim: int = 0,
        rldp_error_dim: int = 0,
        specify_obs_keys: tuple[str, ...] = ("intrinsic_reward", "prev_action"),
        d_hidden: int = 256,
        d_output: int = 256,
        n_layers: int = 2,
        auto_scale: bool = True,
    ):
        super().__init__()
        keys = tuple(specify_obs_keys)
        unknown = [k for k in keys if k not in _POLICY_TSTEP_ENCODER_ALLOWED_KEYS]
        if unknown:
            raise ValueError(
                f"PolicyTstepEncoder.specify_obs_keys got unsupported keys "
                f"{unknown}; allowed: "
                f"{sorted(_POLICY_TSTEP_ENCODER_ALLOWED_KEYS)}."
            )
        self._obs_keys = frozenset(keys)
        self._idm_dim = idm_dim

        if "rldp" in self._obs_keys and rldp_dim <= 0:
            raise ValueError(
                "PolicyTstepEncoder.specify_obs_keys includes 'rldp' but "
                f"rldp_dim={rldp_dim}; pass a positive rldp_dim from the "
                "construction site (TDJepa pulls it from obs_space['rldp'])."
            )
        if "idm" in self._obs_keys and idm_dim <= 0:
            raise ValueError(
                "PolicyTstepEncoder.specify_obs_keys includes 'idm' but "
                f"idm_dim={idm_dim}; pass a positive idm_dim from the "
                "construction site (TDJepa pulls it from obs_space['idm'])."
            )
        if "observed_normalized" in self._obs_keys and observed_normalized_dim <= 0:
            raise ValueError(
                "PolicyTstepEncoder.specify_obs_keys includes "
                "'observed_normalized' but "
                f"observed_normalized_dim={observed_normalized_dim}; pass a "
                "positive observed_normalized_dim from the construction site "
                "(TDJepa pulls it from obs_space['observed_normalized'])."
            )
        if "rldp_error" in self._obs_keys and rldp_error_dim <= 0:
            raise ValueError(
                "PolicyTstepEncoder.specify_obs_keys includes 'rldp_error' "
                f"but rldp_error_dim={rldp_error_dim}; pass a positive "
                "rldp_error_dim from the construction site (TDJepa pulls it "
                "from obs_space['rldp_error'])."
            )

        in_dim = phi_dim
        if "intrinsic_reward" in self._obs_keys:
            in_dim += 1
        if "prev_action" in self._obs_keys:
            in_dim += prev_action_dim
        if "rldp" in self._obs_keys:
            in_dim += rldp_dim
        if "idm" in self._obs_keys:
            in_dim += idm_dim
        if "observed_normalized" in self._obs_keys:
            in_dim += observed_normalized_dim
        if "rldp_error" in self._obs_keys:
            in_dim += rldp_error_dim
        self._in_dim = in_dim

        if auto_scale:
            d_hidden = max(d_hidden, in_dim)
            d_output = max(d_output, in_dim)

        self.base = ff.MLP(
            d_inp=in_dim,
            d_hidden=d_hidden,
            n_layers=n_layers,
            d_output=d_output,
            activation="leaky_relu",
            normalization=None,
        )
        self.out_norm = ff.Normalization("layer", d_output)
        self._emb_dim = d_output

    @property
    def emb_dim(self) -> int:
        return self._emb_dim

    @property
    def in_dim(self) -> int:
        return self._in_dim

    @property
    def obs_keys(self) -> tuple[str, ...]:
        """Return the concat-ordered tuple of active obs keys."""
        return tuple(k for k in self._CONCAT_ORDER if k in self._obs_keys)

    def forward(
        self,
        phi: torch.Tensor,
        obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        missing = [k for k in self._obs_keys if k not in obs]
        assert not missing, (
            f"PolicyTstepEncoder.forward: obs dict is missing required keys "
            f"{missing} (configured via specify_obs_keys="
            f"{sorted(self._obs_keys)}).  Present keys: {sorted(obs.keys())}."
        )
        parts = [phi]
        if "intrinsic_reward" in self._obs_keys:
            parts.append(obs["intrinsic_reward"])
        if "prev_action" in self._obs_keys:
            parts.append(obs["prev_action"].float())
        if "rldp" in self._obs_keys:
            parts.append(obs["rldp"].float())
        if "idm" in self._obs_keys:
            parts.append(obs["idm"].float())
        if "observed_normalized" in self._obs_keys:
            parts.append(obs["observed_normalized"].float())
        if "rldp_error" in self._obs_keys:
            parts.append(obs["rldp_error"].float())
        x = torch.cat(parts, dim=-1)
        return self.out_norm(self.base(x))
