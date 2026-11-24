"""
DynamicsLearner: standalone RLDP dynamics encoder as an AMAGO BaseAgent.

Processes raw observations through tstep_encoder -> DynamicsEncoder to produce
RLDP(s) features. Trained via LDP (Latent Dynamics Prediction) loss jointly
with the RL agent, and augments the batch with detached 'rldp' features for
it. ``NoOpDynamicsLearner`` replaces it for baselines without RLDP.
"""

from __future__ import annotations

import copy
import itertools
from typing import Optional, Type, Tuple, Any

import gin
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym

from amago import utils
from amago.loading import Batch, MAGIC_PAD_VAL
from amago.agent import BaseAgent
from amago.nets import TstepEncoder, TrajEncoder
from amago.nets.traj_encoders import (
    TformerTrajEncoder,
    register_traj_encoder,
)
from amago.nets.utils import add_activation_log


@torch.compile
def _cross_traj_mean_cosine(
    h_masked: torch.Tensor, n_cross: torch.Tensor
) -> torch.Tensor:
    """Mean cosine similarity across trajectories (RLDP Eq 7).

    Uses identity: sum_{i!=j} h_i . h_j = ||sum_i h_i||^2 - sum_i ||h_i||^2.
    """
    h_sum_per_traj = h_masked.sum(dim=1)
    total_dot = (h_sum_per_traj.sum(dim=0) ** 2).sum()
    within_dot = (h_sum_per_traj**2).sum()
    return (total_dot - within_dot) / n_cross


def sanitize_published(
    key: str,
    x: torch.Tensor,
    log_step: bool = False,
    update_info: Optional[dict] = None,
) -> torch.Tensor:
    """nan_to_num + detach for any tensor going into batch.obs.

    Replaces NaN, +inf, -inf with 0.0. Zero is benign for every published
    feature: hypersphere-normalized (rldp, idm, dyn_tstep_emb), bounded
    [0, 2] (rldp_error from cosine distance), or RMS-normalized
    (intrinsic_reward). When ``log_step`` is True and ``update_info`` is
    given, records pre-sanitization counts under
    ``Sanitize/{key}_{nan,inf,nan_frac}``.
    """
    if log_step and update_info is not None:
        n_nan = torch.isnan(x).sum()
        n_inf = torch.isinf(x).sum()
        n_total = max(x.numel(), 1)
        update_info[f"Sanitize/{key}_nan_count"] = n_nan
        update_info[f"Sanitize/{key}_inf_count"] = n_inf
        update_info[f"Sanitize/{key}_nan_frac"] = n_nan.float() / n_total
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).detach()


# ═══════════════════════════════════════════════════════════════════════════
#  DynamicsEncoder — produces RLDP(s), always detached
# ═══════════════════════════════════════════════════════════════════════════


@gin.configurable
@register_traj_encoder("rldp")
class DynamicsEncoder(TrajEncoder):
    """Trajectory encoder producing RLDP(s) via a dynamics backbone.

    The returned s_rep is always detached. Live gradients for the LDP loss
    are available through _cached_h.
    """

    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        dynamics_encoder_type: Type[TrajEncoder] = TformerTrajEncoder,
        sphere_normalize: bool = True,
        dynamics_output_dim: Optional[int] = None,
        idm_dim: Optional[int] = None,
        idm_sphere_normalize: bool = True,
    ):
        super().__init__(tstep_dim, max_seq_len)
        self.dynamics_encoder = dynamics_encoder_type(
            tstep_dim=tstep_dim,
            max_seq_len=max_seq_len,
        )
        raw_dyn_dim = self.dynamics_encoder.emb_dim
        if dynamics_output_dim is not None and dynamics_output_dim != raw_dyn_dim:
            self.output_proj = nn.Linear(raw_dyn_dim, dynamics_output_dim)
            self._dynamics_out_dim = dynamics_output_dim
        else:
            self.output_proj = None
            self._dynamics_out_dim = raw_dyn_dim

        self.sphere_normalize = sphere_normalize
        self._cached_h: Optional[torch.Tensor] = None
        self._cached_s_rep: Optional[torch.Tensor] = None

        # IDM projection head (independent of the RLDP path above)
        self._idm_dim = idm_dim or 0
        self.idm_sphere_normalize = idm_sphere_normalize
        if idm_dim is not None and idm_dim > 0:
            self.idm_proj = nn.Linear(raw_dyn_dim, idm_dim)
        else:
            self.idm_proj = None
        self._cached_idm: Optional[torch.Tensor] = None

        if self.sphere_normalize:
            self._strip_dynamics_out_norm()

    def _strip_dynamics_out_norm(self):
        enc = self.dynamics_encoder
        if hasattr(enc, "out_norm"):
            enc.out_norm = nn.Identity()
        if hasattr(enc, "tformer") and hasattr(enc.tformer, "norm"):
            enc.tformer.norm = nn.Identity()

    @property
    def emb_dim(self) -> int:
        return self._dynamics_out_dim

    @property
    def dynamics_out_dim(self) -> int:
        return self._dynamics_out_dim

    @property
    def idm_dim(self) -> int:
        return self._idm_dim

    def init_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Any]:
        return self.dynamics_encoder.init_hidden_state(batch_size, device)

    def reset_hidden_state(
        self, hidden_state: Optional[Any], dones: np.ndarray
    ) -> Optional[Any]:
        if hidden_state is None:
            return None
        return self.dynamics_encoder.reset_hidden_state(hidden_state, dones)

    def forward(
        self,
        seq: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: Optional[Any] = None,
        log_dict: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        add_activation_log("DynEncoder-input", seq, log_dict)
        h_backbone, new_hidden = self.dynamics_encoder(
            seq, time_idxs, hidden_state, log_dict
        )
        add_activation_log("DynEncoder-backbone-out", h_backbone, log_dict)

        # RLDP path
        h = h_backbone
        if self.output_proj is not None:
            h = self.output_proj(h)
            add_activation_log("DynEncoder-proj-out", h, log_dict)
        if self.sphere_normalize:
            h = F.normalize(h, dim=-1) * (h.shape[-1] ** 0.5)
        add_activation_log("DynEncoder-out", h, log_dict)
        self._cached_h = h
        s_rep = h.detach()
        self._cached_s_rep = s_rep

        # IDM path (independent projection from the shared backbone)
        if self.idm_proj is not None:
            idm = self.idm_proj(h_backbone)
            if self.idm_sphere_normalize:
                idm = F.normalize(idm, dim=-1) * (idm.shape[-1] ** 0.5)
            self._cached_idm = idm
        else:
            self._cached_idm = None

        return s_rep, new_hidden


# ═══════════════════════════════════════════════════════════════════════════
#  DynamicsLearner — standalone BaseAgent for RLDP dynamics
# ═══════════════════════════════════════════════════════════════════════════


@gin.configurable
class DynamicsLearner(BaseAgent):
    """Standalone RLDP dynamics encoder trained via LDP loss.

    Processes raw observations, produces detached RLDP features, and augments
    the batch with batch.obs['rldp']. Trained via Latent Dynamics Prediction
    loss plus diagnostic observation probes (nonlinear MSE regression of the
    full ``obs["observed"]`` vector from current and next-step RLDP features).
    """

    def __init__(
        self,
        obs_space: gym.spaces.Dict,
        rl2_space: gym.spaces.Box,
        action_space: gym.spaces.Space,
        max_seq_len: int,
        tstep_encoder_type: Type[TstepEncoder] = gin.REQUIRED,
        traj_encoder_type: Type[TrajEncoder] = DynamicsEncoder,
        # LDP config
        ldp_coeff: float = 1.0,
        ldp_ortho_coeff: float = 0.01,
        ldp_target: str = "tstep",
        ldp_g_hidden: int = 512,
        ldp_g_n_hidden: int = 2,
        ldp_ema_tau: float = 0.005,
        detach_probes: bool = True,
        probe_hidden: int = 400,
        # IDM config
        idm_coeff: float = 0.0,
        idm_hidden: int = 400,
        # Skip-connection: feed previous timestep's target into prediction heads
        skip_prev_target: bool = False,
    ):
        super().__init__(
            obs_space=obs_space,
            rl2_space=rl2_space,
            action_space=action_space,
            max_seq_len=max_seq_len,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
        )
        # rl2s (= [prev_action, prev_reward]) is hidden from every encoder in
        # this codebase.  Set before the target_tstep_encoder deepcopy below
        # so the EMA copy inherits it.
        self.tstep_encoder.hide_rl2s = True
        self.tau = ldp_ema_tau
        self.ldp_coeff = ldp_coeff
        self.ldp_ortho_coeff = ldp_ortho_coeff
        assert ldp_target in ("tstep", "hidden")
        self.ldp_target = ldp_target
        self.detach_probes = detach_probes
        self.skip_prev_target = skip_prev_target
        self.update_info: dict = {}
        self._cached_ldp_cos_sim = None
        self._cached_tstep_emb: Optional[torch.Tensor] = None

        h_dim = self.traj_encoder.dynamics_out_dim
        s_rep_dim = self.traj_encoder.emb_dim

        # Apply sphere normalization to tstep encoder if dynamics uses it
        if self.traj_encoder.sphere_normalize:
            from amago.nets.ff import Normalization

            self.tstep_encoder.out_norm = Normalization(
                "hypersphere", self.tstep_encoder.emb_dim
            )

        # LDP prediction heads: g predicts dynamics, w projects to target space
        target_dim = self.tstep_encoder.emb_dim if ldp_target == "tstep" else h_dim
        g_hidden = ldp_g_hidden
        g_in_dim = h_dim + self.action_dim
        if skip_prev_target:
            g_in_dim += target_dim
        g_layers = [
            nn.Linear(g_in_dim, g_hidden),
            nn.LayerNorm(g_hidden),
            nn.LeakyReLU(),
        ]
        for _ in range(ldp_g_n_hidden - 1):
            g_layers += [
                nn.Linear(g_hidden, g_hidden),
                nn.LayerNorm(g_hidden),
                nn.LeakyReLU(),
            ]
        g_layers.append(nn.Linear(g_hidden, h_dim))
        self.dynamics_g = nn.Sequential(*g_layers)
        self.dynamics_w = nn.Linear(h_dim, target_dim, bias=False)

        # EMA target tstep encoder for LDP targets
        self.target_tstep_encoder = copy.deepcopy(self.tstep_encoder)
        self.target_tstep_encoder.requires_grad_(False)

        # Observation probes (diagnostic when detach_probes=True; otherwise
        # they also train the dynamics encoder).
        obs_dim = obs_space["observed"].shape[-1]
        next_in_dim = s_rep_dim + self.action_dim
        if skip_prev_target:
            next_in_dim += obs_dim
        self.obs_probe_current = nn.Sequential(
            nn.Linear(s_rep_dim, probe_hidden),
            nn.LeakyReLU(),
            nn.Linear(probe_hidden, probe_hidden),
            nn.LeakyReLU(),
            nn.Linear(probe_hidden, obs_dim),
        )
        self.obs_probe_next = nn.Sequential(
            nn.Linear(next_in_dim, probe_hidden),
            nn.LeakyReLU(),
            nn.Linear(probe_hidden, probe_hidden),
            nn.LeakyReLU(),
            nn.Linear(probe_hidden, obs_dim),
        )
        # Probe targets are the per-dim z-scored obs["observed"], published as
        # batch.obs["observed_normalized"] by BilevelAgent. MSE ~1.0 is
        # chance, ~0 is perfect reconstruction.

        # IDM head: predict a_t from (idm_t, idm_{t+1}).
        self.idm_coeff = idm_coeff
        idm_dim = self.traj_encoder.idm_dim
        if idm_coeff > 0:
            assert (
                idm_dim > 0
            ), "idm_coeff > 0 requires DynamicsEncoder.idm_dim to be set"
            assert (
                not self.discrete
            ), "IDM currently supports continuous action spaces only"
            idm_in_dim = 2 * idm_dim
            if skip_prev_target:
                idm_in_dim += 2 * self.tstep_encoder.emb_dim
            self.idm_head = nn.Sequential(
                nn.Linear(idm_in_dim, idm_hidden),
                nn.LeakyReLU(),
                nn.Linear(idm_hidden, idm_hidden),
                nn.LeakyReLU(),
                nn.Linear(idm_hidden, self.action_dim),
                nn.Tanh(),
            )
        else:
            self.idm_head = None

        self.hard_sync_targets()

    # ------------------------------------------------------------------
    #  Encode: produce RLDP features from raw observations
    # ------------------------------------------------------------------

    def encode(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """Run dynamics pipeline and return detached RLDP features."""
        tstep_emb = self.tstep_encoder(obs=obs, rl2s=rl2s)
        self._cached_tstep_emb = tstep_emb
        s_rep, hidden_state = self.traj_encoder(
            tstep_emb, time_idxs=time_idxs, hidden_state=hidden_state
        )
        return s_rep.detach(), hidden_state

    # ------------------------------------------------------------------
    #  Inference: prediction error from previous step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_prediction_error(
        self,
        prev_rldp: torch.Tensor,
        prev_action: torch.Tensor,
        current_obs: dict[str, torch.Tensor],
        current_rl2s: torch.Tensor,
        prev_online: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute LDP prediction error for a single inference timestep.

        Matches the cosine distance in ``_compute_ldp_loss``
        (ldp_target="tstep") exactly: same normalization, same eps.

        ``prev_online`` is the previous step's *online* tstep encoder output
        (``_cached_tstep_emb``), as in the training-time skip input. The
        target is the EMA target tstep encoder on ``current_obs``.

        Returns zeros when ``ldp_coeff == 0``, as ``forward`` does: the
        untrained ``dynamics_g`` / ``dynamics_w`` would otherwise publish
        meaningless errors to any ``rldp_error`` consumer.

        Returns:
            (..., 1) tensor of prediction errors with the same leading shape
            as ``prev_rldp`` minus its feature dim.
        """
        if self.ldp_coeff == 0:
            return torch.zeros(
                *prev_rldp.shape[:-1],
                1,
                device=prev_rldp.device,
                dtype=prev_rldp.dtype,
            )
        assert (
            self.ldp_target == "tstep"
        ), f"rldp_error requires ldp_target='tstep', got '{self.ldp_target}'"
        a = prev_action.clamp(0, 1.0) if self.discrete else prev_action.clamp(-1.0, 1.0)
        g_input = [prev_rldp, a]
        if self.skip_prev_target:
            assert (
                prev_online is not None
            ), "skip_prev_target=True requires prev_online to be provided"
            g_input.append(prev_online)
        g_out = self.dynamics_g(torch.cat(g_input, dim=-1))
        pred = self.dynamics_w(g_out)
        if self.traj_encoder.sphere_normalize:
            pred = F.normalize(pred, dim=-1) * (pred.shape[-1] ** 0.5)
        target = self.target_tstep_encoder(obs=current_obs, rl2s=current_rl2s)
        cos_sim = F.cosine_similarity(pred, target, dim=-1, eps=1e-6)
        return (1.0 - cos_sim).unsqueeze(-1)

    # ------------------------------------------------------------------
    #  Forward: training pass (augments batch, returns loss)
    # ------------------------------------------------------------------

    def forward(self, batch: Batch, log_step: bool) -> torch.Tensor:
        self.update_info = {}
        active_log_dict = {} if log_step else None

        # Run dynamics pipeline (populates traj_encoder._cached_h).
        # Cache the live (gradient-attached) tstep emb so the IDM and LDP skip
        # inputs backprop into the online tstep encoder; dyn_tstep_emb is
        # published downstream detached.
        o = self.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
        self._cached_tstep_emb = o
        add_activation_log("DynLearner-tstep-out", o, active_log_dict)
        s_rep, _ = self.traj_encoder(
            seq=o,
            time_idxs=batch.time_idxs,
            hidden_state=None,
            log_dict=active_log_dict,
        )
        if log_step:
            self.update_info["SeqLen/Dynamics"] = o.shape[1]

        # Augment batch with detached, NaN-sanitized features (see
        # ``sanitize_published``).
        batch.obs["rldp"] = sanitize_published(
            "rldp", s_rep, log_step=log_step, update_info=self.update_info
        )
        batch.obs["dyn_tstep_emb"] = sanitize_published(
            "dyn_tstep_emb", o, log_step=log_step, update_info=self.update_info
        )

        cached_idm = self.traj_encoder._cached_idm
        if cached_idm is not None:
            batch.obs["idm"] = sanitize_published(
                "idm", cached_idm, log_step=log_step, update_info=self.update_info
            )
        else:
            B, L = s_rep.shape[:2]
            batch.obs["idm"] = torch.zeros(B, L, 1, device=s_rep.device)
        if log_step and cached_idm is not None:
            with torch.no_grad():
                idm_feat = cached_idm.detach()
                pad_val = MAGIC_PAD_VAL
                full_mask = (~((batch.rl2s == pad_val).all(-1))).float()
                state_mask = full_mask[:, 1:]  # (B, L-1)
                mask_flat = state_mask.reshape(-1).bool()

                D_idm = idm_feat.shape[-1]
                idm_valid = idm_feat[:, 1:].reshape(-1, D_idm)[mask_flat]

                self.update_info["IDM/Obs Mean"] = idm_feat.mean()

                if idm_valid.shape[0] > 1:
                    idm_normed = F.normalize(idm_valid, dim=-1)
                    sim_mat = idm_normed @ idm_normed.T
                    n = idm_valid.shape[0]
                    avg_cos = (sim_mat.sum() - n) / (n * (n - 1))
                    self.update_info["IDM/Avg Cosine Sim"] = avg_cos

                    feat_std = idm_valid.std(dim=0)
                    self.update_info["IDM/Avg Feature Std"] = feat_std.mean()
                    self.update_info["IDM/Min Feature Std"] = feat_std.min()
                    self.update_info["IDM/Dead Features"] = (feat_std < 1e-4).sum()

                    # Within-sequence adjacent cosine sim
                    idm_seq_norm = F.normalize(idm_feat[:, 1:], dim=-1)
                    adj_cos = (idm_seq_norm[:, :-1] * idm_seq_norm[:, 1:]).sum(dim=-1)
                    adj_mask = state_mask[:, :-1] * state_mask[:, 1:]
                    adj_denom = adj_mask.sum().clamp(min=1)
                    self.update_info["IDM/Adjacent Cosine Sim"] = (
                        adj_cos * adj_mask
                    ).sum() / adj_denom

        # Compute losses
        total = torch.zeros(1, device=s_rep.device).squeeze()
        if self.ldp_coeff > 0:
            total = total + self._compute_ldp_loss(batch, log_step)
        if self.idm_coeff > 0:
            total = total + self._compute_idm_loss(batch, log_step)

        # h-feature diagnostics: fire whenever _cached_h is populated,
        # regardless of ldp_coeff, so Rldp/* metrics exist with LDP on or off.
        if log_step and self.traj_encoder._cached_h is not None:
            self._log_h_diagnostics(batch)

        # Attach per-timestep LDP prediction error (shifted by one so position
        # t contains the error from predicting target_t using [rldp_{t-1}, a_{t-1}]).
        # Inert unless an agent's specify_obs_keys includes "rldp_error".
        if self._cached_ldp_cos_sim is not None:
            error = 1.0 - self._cached_ldp_cos_sim
            B_err = error.shape[0]
            prev_error = torch.cat(
                [torch.zeros(B_err, 1, device=error.device), error], dim=1
            )
            batch.obs["rldp_error"] = sanitize_published(
                "rldp_error",
                prev_error.unsqueeze(-1),
                log_step=log_step,
                update_info=self.update_info,
            )
            if log_step:
                self.update_info.update(
                    {
                        "LDP/RLDP Error Mean": error.mean(),
                        "LDP/RLDP Error Min": error.min(),
                        "LDP/RLDP Error Max": error.max(),
                    }
                )
            self._cached_ldp_cos_sim = None
        elif "rldp_error" not in batch.obs:
            B, L = s_rep.shape[:2]
            batch.obs["rldp_error"] = torch.zeros(B, L, 1, device=s_rep.device)

        total = total + self._compute_obs_probe_loss(batch, log_step)
        if active_log_dict:
            self.update_info.update(active_log_dict)
        return total

    # ------------------------------------------------------------------
    #  get_actions (BaseAgent interface — not used in practice)
    # ------------------------------------------------------------------

    def get_actions(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state=None,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError(
            "DynamicsLearner does not produce actions. "
            "Use BilevelAgent for inference."
        )

    # ------------------------------------------------------------------
    #  LDP loss
    # ------------------------------------------------------------------

    def _compute_ldp_loss(self, batch: Batch, log_step: bool) -> torch.Tensor:
        log_dict = {} if log_step else None
        h = self.traj_encoder._cached_h  # (B, L, D_h)
        B, L, D_h = h.shape
        a = batch.actions
        a = a.clamp(0, 1.0) if self.discrete else a.clamp(-1.0, 1.0)

        h_t = h[:, :-1]

        if self.ldp_target == "tstep":
            with torch.no_grad():
                target_full = self.target_tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
            target = target_full[:, 1:]
        else:
            target_full = h.detach()
            target = target_full[:, 1:]
        add_activation_log("LDP-target", target, log_dict)

        if self.skip_prev_target:
            online_prev = self._cached_tstep_emb[:, :-1]  # live, gradient-attached
            g_out = self.dynamics_g(torch.cat([h_t, a, online_prev], dim=-1))
        else:
            g_out = self.dynamics_g(torch.cat([h_t, a], dim=-1))
        add_activation_log("LDP-g-out", g_out, log_dict)
        pred = self.dynamics_w(g_out)
        add_activation_log("LDP-w-out", pred, log_dict)

        if self.traj_encoder.sphere_normalize:
            if self.ldp_target == "tstep":
                pred = F.normalize(pred, dim=-1) * (pred.shape[-1] ** 0.5)
            else:
                pred = F.normalize(pred, dim=-1) * (D_h**0.5)
        add_activation_log("LDP-pred-final", pred, log_dict)

        state_mask = (~(batch.rl2s == self.pad_val).all(-1, keepdim=True)).float()[
            :, 1:
        ]

        if self.ldp_target == "tstep":
            cos_sim = F.cosine_similarity(pred, target, dim=-1, eps=1e-6)
            self._cached_ldp_cos_sim = cos_sim.detach()
            ldp_loss = (1.0 - cos_sim).unsqueeze(-1)
        else:
            self._cached_ldp_cos_sim = None
            ldp_loss = (pred - target).pow(2).mean(-1, keepdim=True)
        ldp_loss = (ldp_loss * state_mask).sum() / state_mask.sum().clamp(min=1)

        # Cross-traj mean cosine for the ortho loss. The h-feature diagnostics
        # live in _log_h_diagnostics.
        ortho_loss = torch.zeros(1, device=h.device).squeeze()
        if self.ldp_ortho_coeff > 0:
            h_post_norm = F.normalize(h[:, 1:], dim=-1)
            h_masked = h_post_norm * state_mask
            valid_per_traj = state_mask.squeeze(-1).sum(dim=1)
            n_valid_trajs = (valid_per_traj > 0).sum()
            assert n_valid_trajs > 1, (
                f"LDP ortho requires >= 2 trajectories with valid data, "
                f"got {n_valid_trajs}"
            )
            n_valid = valid_per_traj.sum()
            n_cross = n_valid**2 - (valid_per_traj**2).sum()
            if n_cross > 0:
                ortho_loss = _cross_traj_mean_cosine(h_masked, n_cross)

        total = self.ldp_coeff * ldp_loss + self.ldp_ortho_coeff * ortho_loss

        if log_step:
            with torch.no_grad():
                pred_cos = F.cosine_similarity(pred, target, dim=-1)
                pred_cos_masked = (
                    pred_cos * state_mask.squeeze(-1)
                ).sum() / state_mask.sum().clamp(min=1)
            self.update_info.update(
                {
                    "LDP/Prediction Loss": ldp_loss,
                    "LDP/Ortho Loss": ortho_loss,
                    "LDP/Total": total,
                    "LDP/Pred Cosine Sim": pred_cos_masked,
                }
            )
            self.update_info.update(log_dict)
        return total

    # ------------------------------------------------------------------
    #  h-feature diagnostics
    # ------------------------------------------------------------------

    def _log_h_diagnostics(self, batch: Batch) -> None:
        """Log h-feature health metrics independently of the LDP loss path.

        Reads ``self.traj_encoder._cached_h`` and ``batch.rl2s`` and writes
        ``Rldp/...`` keys describing the post-transformer feature h, with
        LDP on or off.
        """
        h = self.traj_encoder._cached_h
        if h is None:
            return
        with torch.no_grad():
            B, L, D_h = h.shape
            state_mask = (~(batch.rl2s == self.pad_val).all(-1, keepdim=True)).float()[
                :, 1:
            ]
            h_post_norm = F.normalize(h[:, 1:], dim=-1)
            h_masked = h_post_norm * state_mask

            valid_per_traj = state_mask.squeeze(-1).sum(dim=1)
            has_valid_seq = valid_per_traj > 0
            n_valid = valid_per_traj.sum()
            n_cross = n_valid**2 - (valid_per_traj**2).sum()

            # Cross-traj avg / max cosine.
            if n_cross > 0:
                avg_cos_sim = _cross_traj_mean_cosine(h_masked, n_cross).detach()
            else:
                avg_cos_sim = torch.zeros(1, device=h.device).squeeze()

            n_vt = has_valid_seq.sum()
            if n_vt > 1:
                sampled_t = torch.multinomial(
                    state_mask.squeeze(-1)[has_valid_seq], 1
                ).squeeze(1)
                b_idx = torch.arange(n_vt, device=h.device)
                h_sample = h_post_norm[has_valid_seq][b_idx, sampled_t]
                sg = h_sample @ h_sample.T
                eye = torch.eye(n_vt, device=h.device, dtype=torch.bool)
                max_cos_sim = sg[~eye].max()
            else:
                max_cos_sim = torch.zeros(1, device=h.device).squeeze()

            mask_l = state_mask.squeeze(-1)  # (B, L-1)

            # Within-Seq Avg Cosine Sim via the per-traj sum-of-vectors trick:
            # sum_{t1!=t2} h_b[t1]·h_b[t2] = ||sum_t h_b[t]||^2 - valid_per_traj[b].
            within_sum_sq = (h_masked.sum(dim=1) ** 2).sum(dim=-1)
            within_pair_dot = within_sum_sq - valid_per_traj
            within_pair_count = valid_per_traj * (valid_per_traj - 1).clamp(min=0)
            has_within_pairs = within_pair_count > 0
            if has_within_pairs.any():
                within_avg_cos_sim = (
                    within_pair_dot[has_within_pairs]
                    / within_pair_count[has_within_pairs]
                ).mean()
            else:
                within_avg_cos_sim = torch.zeros(1, device=h.device).squeeze()

            # Within-Seq Max Cosine Sim: stochastic K-timestep sample per traj.
            K = min(64, h_post_norm.shape[1])
            if has_valid_seq.any() and K > 1:
                sample_probs = mask_l[has_valid_seq] + 1e-10
                idx_sample = torch.multinomial(sample_probs, K, replacement=True)
                h_v = h_post_norm[has_valid_seq]
                b_idx_k = (
                    torch.arange(h_v.shape[0], device=h.device)
                    .unsqueeze(1)
                    .expand(-1, K)
                )
                h_sample_k = h_v[b_idx_k, idx_sample]
                gram_k = torch.bmm(h_sample_k, h_sample_k.transpose(-1, -2))
                eye_k = torch.eye(K, device=h.device, dtype=torch.bool).unsqueeze(0)
                gram_k = gram_k.masked_fill(eye_k, torch.finfo(gram_k.dtype).min)
                within_max_cos_sim = (
                    gram_k.reshape(gram_k.shape[0], -1).max(dim=1).values.mean()
                )
            else:
                within_max_cos_sim = torch.zeros(1, device=h.device).squeeze()

            # Adjacent cosine: cos(h_b[t], h_b[t+1]) averaged over consecutive
            # valid pairs. Indicates whether h actually moves step-by-step.
            adj_cos = (h_post_norm[:, :-1] * h_post_norm[:, 1:]).sum(dim=-1)
            adj_mask = mask_l[:, :-1] * mask_l[:, 1:]
            adj_denom = adj_mask.sum().clamp(min=1)
            adjacent_cos_sim = (adj_cos * adj_mask).sum() / adj_denom

            mask_flat = state_mask.reshape(-1).bool()
            h_valid_raw = h[:, 1:].reshape(-1, D_h)[mask_flat]
            feat_std = h_valid_raw.std(dim=0)
            avg_feat_std = feat_std.mean()
            min_feat_std = feat_std.min()
            dead_features = (feat_std < 1e-4).sum()

            if h_valid_raw.shape[0] > 1:
                s = torch.linalg.svdvals(
                    h_valid_raw - h_valid_raw.mean(dim=0, keepdim=True)
                )
                p = s / s.sum().clamp(min=1e-8)
                eff_rank = torch.exp(-(p * (p + 1e-8).log()).sum())
            else:
                eff_rank = torch.ones(1, device=h.device).squeeze()

        self.update_info.update(
            {
                "Rldp/Avg Cosine Sim": avg_cos_sim,
                "Rldp/Max Cosine Sim": max_cos_sim,
                "Rldp/Within-Seq Avg Cosine Sim": within_avg_cos_sim,
                "Rldp/Within-Seq Max Cosine Sim": within_max_cos_sim,
                "Rldp/Adjacent Cosine Sim": adjacent_cos_sim,
                "Rldp/Avg Feature Std": avg_feat_std,
                "Rldp/Min Feature Std": min_feat_std,
                "Rldp/Dead Features": dead_features,
                "Rldp/Effective Rank": eff_rank,
            }
        )

    # ------------------------------------------------------------------
    #  IDM loss
    # ------------------------------------------------------------------

    def _compute_idm_loss(self, batch: Batch, log_step: bool) -> torch.Tensor:
        """Inverse dynamics prediction: (idm_t, idm_{t+1}) -> a_t (MSE)."""
        idm = self.traj_encoder._cached_idm  # (B, L, idm_dim)
        assert idm is not None, "IDM loss requires DynamicsEncoder.idm_dim > 0"
        # Clamp inside tanh's reachable range. Pure [-1, 1] targets force the
        # idm_head's pre-tanh logits to ±∞ (since tanh(x)=±1 requires x=±∞),
        # which makes the head's weights grow unbounded under MSE and
        # eventually corrupts the dynamics traj_encoder via backprop.
        # arctanh(0.99) ≈ 2.65 is comfortably reachable with finite weights.
        a = batch.actions.clamp(-0.99, 0.99)  # (B, L-1, action_dim)

        if self.skip_prev_target:
            e = self._cached_tstep_emb  # (B, L, tstep_dim), live online encoder output
            idm_input = torch.cat(
                [idm[:, :-1], idm[:, 1:], e[:, :-1], e[:, 1:]], dim=-1
            )
        else:
            idm_input = torch.cat(
                [idm[:, :-1], idm[:, 1:]], dim=-1
            )  # (B, L-1, 2*idm_dim)
        pred_a = self.idm_head(idm_input)  # (B, L-1, action_dim)

        state_mask = (~(batch.rl2s == self.pad_val).all(-1, keepdim=True)).float()[
            :, 1:
        ]
        mse = (pred_a - a).pow(2).mean(-1, keepdim=True)
        idm_loss = (mse * state_mask).sum() / state_mask.sum().clamp(min=1)

        total = self.idm_coeff * idm_loss

        if log_step:
            with torch.no_grad():
                action_mse = (mse * state_mask).sum() / state_mask.sum().clamp(min=1)
            self.update_info.update(
                {
                    "IDM/Loss": total,
                    "IDM/Action MSE": action_mse,
                }
            )
        return total

    # ------------------------------------------------------------------
    #  Observation probes
    # ------------------------------------------------------------------

    def _compute_obs_probe_loss(self, batch: Batch, log_step: bool) -> torch.Tensor:
        """Nonlinear MSE probes regressing the full ``obs["observed"]`` vector.

        Two heads:
          - current: RLDP_t -> observed_t
          - next:    (RLDP_t, action_t) -> observed_{t+1}
        Gradients flow into the dynamics backbone iff ``detach_probes=False``.
        """
        s_rep = (
            self.traj_encoder._cached_s_rep
            if self.detach_probes
            else self.traj_encoder._cached_h
        )
        state_mask = (~(batch.rl2s == self.pad_val).all(-1, keepdim=True)).float()
        next_mask = state_mask[:, 1:]
        assert "observed_normalized" in batch.obs, (
            "DynamicsLearner expects BilevelAgent to publish 'observed_normalized'"
            " into batch.obs before forward."
        )
        obs_norm = batch.obs["observed_normalized"]
        target_next = obs_norm[:, 1:]
        a = batch.actions
        a = a.clamp(0, 1.0) if self.discrete else a.clamp(-1.0, 1.0)
        if self.skip_prev_target:
            obs_prev = obs_norm[:, :-1].detach()
            next_input = torch.cat([s_rep[:, :-1], a, obs_prev], dim=-1)
        else:
            next_input = torch.cat([s_rep[:, :-1], a], dim=-1)

        pred_curr = self.obs_probe_current(s_rep)
        curr_mse = ((pred_curr - obs_norm) ** 2).mean(-1, keepdim=True)
        curr_loss = (curr_mse * state_mask).sum() / state_mask.sum().clamp(min=1)

        pred_next = self.obs_probe_next(next_input)
        next_mse = ((pred_next - target_next) ** 2).mean(-1, keepdim=True)
        next_loss = (next_mse * next_mask).sum() / next_mask.sum().clamp(min=1)

        total = curr_loss + next_loss

        if log_step:
            log_data = {
                "ObsProbe/Current MSE": curr_loss,
                "ObsProbe/Next MSE": next_loss,
            }
            if "episode_id" in batch.obs:
                ep_ids = batch.obs["episode_id"].long()
                if ep_ids.dim() == 3:
                    ep_ids = ep_ids.squeeze(-1)
                unique_eps = sorted(ep_ids.unique().tolist())
                if len(unique_eps) > 1:
                    unique_eps = unique_eps[:-1]
                for ep in unique_eps:
                    ep_label = ep + 1
                    ep_mask_c = (ep_ids == ep).unsqueeze(-1).float() * state_mask
                    nc = ep_mask_c.sum().clamp(min=1)
                    log_data[f"ObsProbe/Current MSE Ep{ep_label}"] = (
                        curr_mse * ep_mask_c
                    ).sum() / nc

                    ep_mask_n = (ep_ids[:, 1:] == ep).unsqueeze(-1).float() * next_mask
                    nn_ = ep_mask_n.sum().clamp(min=1)
                    log_data[f"ObsProbe/Next MSE Ep{ep_label}"] = (
                        next_mse * ep_mask_n
                    ).sum() / nn_
            self.update_info.update(log_data)
        return total

    # ------------------------------------------------------------------
    #  Training infrastructure
    # ------------------------------------------------------------------

    @property
    def trainable_params(self):
        chains = [
            self.tstep_encoder.parameters(),
            self.traj_encoder.parameters(),
            self.dynamics_g.parameters(),
            self.dynamics_w.parameters(),
            self.obs_probe_current.parameters(),
            self.obs_probe_next.parameters(),
        ]
        if self.idm_head is not None:
            chains.append(self.idm_head.parameters())
        return itertools.chain(*chains)

    def hard_sync_targets(self):
        self._full_copy(self.target_tstep_encoder, self.tstep_encoder)

    def soft_sync_targets(self):
        self._ema_copy(self.target_tstep_encoder, self.tstep_encoder)

    def get_grad_norms(self) -> dict[str, float]:
        norms = {
            "Dynamics/TstepEncoder Grad Norm": utils.get_grad_norm(self.tstep_encoder),
            "Dynamics/TrajEncoder Grad Norm": utils.get_grad_norm(self.traj_encoder),
            "Dynamics/DynamicsEncoder Grad Norm": utils.get_grad_norm(
                self.traj_encoder.dynamics_encoder
            ),
            "Dynamics/G Grad Norm": utils.get_grad_norm(self.dynamics_g),
            "Dynamics/W Grad Norm": utils.get_grad_norm(self.dynamics_w),
        }
        if self.idm_head is not None:
            norms["Dynamics/IDM Head Grad Norm"] = utils.get_grad_norm(self.idm_head)
        return norms

    @property
    def rldp_dim(self) -> int:
        """Output dimension of the RLDP features."""
        return self.traj_encoder.dynamics_out_dim

    @property
    def idm_dim(self) -> int:
        """Output dimension of the IDM features (0 when IDM is disabled)."""
        return self.traj_encoder.idm_dim

    @property
    def dyn_tstep_dim(self) -> int:
        """Output dimension of the dynamics tstep encoder embedding."""
        return self.tstep_encoder.emb_dim


# ═══════════════════════════════════════════════════════════════════════════
#  NoOpDynamicsLearner — passthrough for non-RLDP baselines
# ═══════════════════════════════════════════════════════════════════════════


@gin.configurable
class NoOpDynamicsLearner(nn.Module):
    """No-op dynamics learner that writes a scalar-zero 'rldp' key.

    Used for non-RLDP baselines so the observation space is uniform: every
    agent always sees an 'rldp' key, it just carries no information here.

    Does NOT inherit from BaseAgent — no encoders are constructed.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Dict,
        rl2_space: gym.spaces.Box,
        action_space: gym.spaces.Space,
        max_seq_len: int,
        tstep_encoder_type: Type[TstepEncoder] = None,
        traj_encoder_type: Type[TrajEncoder] = None,
    ):
        super().__init__()
        self.skip_prev_target = False
        self.update_info: dict = {}
        self._stub_traj_encoder = _StubTrajEncoder()
        self._cached_tstep_emb: Optional[torch.Tensor] = None

    def forward(self, batch: Batch, log_step: bool) -> torch.Tensor:
        self.update_info = {}
        B, L = batch.rl2s.shape[:2]
        batch.obs["rldp"] = torch.zeros(B, L, 1, device=batch.rl2s.device)
        batch.obs["rldp_error"] = torch.zeros(B, L, 1, device=batch.rl2s.device)
        batch.obs["idm"] = torch.zeros(B, L, 1, device=batch.rl2s.device)
        batch.obs["dyn_tstep_emb"] = torch.zeros(B, L, 1, device=batch.rl2s.device)
        return torch.zeros(1, device=batch.rl2s.device).squeeze()

    def encode(self, obs, rl2s, time_idxs, hidden_state=None):
        B, L = rl2s.shape[:2]
        self._cached_tstep_emb = torch.zeros(B, L, 1, device=rl2s.device)
        return torch.zeros(B, L, 1, device=rl2s.device), hidden_state

    def compute_prediction_error(
        self,
        prev_rldp: torch.Tensor,
        prev_action: torch.Tensor,
        current_obs: dict[str, torch.Tensor],
        current_rl2s: torch.Tensor,
        prev_online: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = prev_rldp.shape[0]
        return torch.zeros(B, 1, 1, device=prev_rldp.device)

    def get_actions(self, obs, rl2s, time_idxs, hidden_state=None, sample=True):
        raise NotImplementedError("NoOpDynamicsLearner does not produce actions.")

    @property
    def traj_encoder(self):
        return self._stub_traj_encoder

    @property
    def trainable_params(self):
        return iter([])

    def hard_sync_targets(self):
        pass

    def soft_sync_targets(self):
        pass

    def get_grad_norms(self) -> dict[str, float]:
        return {}

    @property
    def rldp_dim(self) -> int:
        return 1

    @property
    def idm_dim(self) -> int:
        return 1

    @property
    def dyn_tstep_dim(self) -> int:
        return 1


class _StubTrajEncoder:
    """Minimal stub satisfying the hidden-state protocol for BilevelAgent."""

    def __init__(self):
        self.emb_dim = 0
        self._cached_idm = None

    def init_hidden_state(self, batch_size, device):
        return None

    def reset_hidden_state(self, hidden_state, dones):
        return None
