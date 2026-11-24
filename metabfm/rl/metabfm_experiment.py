"""BilevelAgent and BilevelExperiment: DynamicsLearner + RLAgent training stack.

Launch scripts (e.g. ``scripts/train_colorgrid.py``) instantiate
:class:`BilevelExperiment`; this module defines the shared agent wrapper,
training loop, and evaluation hooks used by all such launchers.
"""

from __future__ import annotations

import os

from typing import Optional, Tuple, Any, Iterable

import gin
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from accelerate.utils import tqdm

from amago import Experiment, utils
from amago.envs.amago_env import SequenceWrapper
from amago.envs.env_utils import DummyAsyncVectorEnv
from amago.loading import (
    Batch,
    RLData_pad_collate,
    MAGIC_PAD_VAL,
)
from amago.agent import BaseAgent
from amago.nets.utils import InputNorm

import metabfm.rl.zero_shot_rl  # noqa: F401  register gin configurables before lock
import metabfm.rl.dynamics_learner  # noqa: F401  register DynamicsLearner + DynamicsEncoder
from metabfm.rl.dynamics_learner import (
    DynamicsLearner,
    NoOpDynamicsLearner,
    sanitize_published,
)
from metabfm.rl.zero_shot_rl import ZeroShotAgent


# ──────────────────────────────────────────────────────────────────────────────
# Bilevel Agent
# ──────────────────────────────────────────────────────────────────────────────


class _BilevelHiddenState:
    """Bundles the RL agent's and dynamics learner's hidden states.

    The experiment loop manages hidden state via policy.traj_encoder, which
    returns a _BilevelTrajEncoderProxy.  That proxy creates/resets instances
    of this class so both the RL encoder and the dynamics transformer KV-cache
    stay in sync.
    """

    __slots__ = (
        "rl_hidden",
        "dyn_hidden",
        "prev_rldp",
        "prev_action",
        "prev_valid",
        "e3b_sigma_inv",
        "ngu_mem",
        "ngu_step_idx",
        "prev_episode_id",
        "prev_online",
    )

    def __init__(
        self,
        rl_hidden,
        dyn_hidden,
        prev_rldp=None,
        prev_action=None,
        prev_valid=None,
        e3b_sigma_inv=None,
        ngu_mem=None,
        ngu_step_idx=None,
        prev_episode_id=None,
        prev_online=None,
    ):
        self.rl_hidden = rl_hidden
        self.dyn_hidden = dyn_hidden
        self.prev_rldp = prev_rldp
        self.prev_action = prev_action
        self.prev_valid = prev_valid
        self.e3b_sigma_inv = e3b_sigma_inv
        self.ngu_mem = ngu_mem
        self.ngu_step_idx = ngu_step_idx
        self.prev_episode_id = prev_episode_id
        self.prev_online = prev_online


class _BilevelTrajEncoderProxy:
    """Proxy returned by BilevelAgent.traj_encoder.

    Forwards attribute access to the RL agent's primary traj_encoder but
    intercepts init_hidden_state / reset_hidden_state to manage *all*
    hidden states: the dynamics learner's KV-cache AND the RL agent's
    own (which may be a tuple for dual-backbone agents like TDJepa).
    """

    def __init__(self, rl_agent: "BaseAgent", dyn_traj_encoder):
        self._rl_agent = rl_agent
        self._dyn = dyn_traj_encoder

    def __getattr__(self, name):
        return getattr(self._rl_agent.traj_encoder, name)

    def _init_e3b_sigma_inv(self, batch_size: int, device) -> Optional[torch.Tensor]:
        """Allocate (1/λ) I per row when E3B is on; otherwise return None."""
        intrinsic_stack = getattr(self._rl_agent, "intrinsic_stack", None)
        if intrinsic_stack is None or intrinsic_stack.e3b_coeff <= 0:
            return None
        d = intrinsic_stack.e3b_feature_dim
        lam = intrinsic_stack.e3b_lambda
        eye = torch.eye(d, device=device) / lam
        return eye.unsqueeze(0).expand(batch_size, d, d).contiguous()

    def _init_ngu_state(
        self, batch_size: int, device
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Allocate (B, max_seq_len, d_f) zero memory and (B,) step counter
        when NGU is on; otherwise return (None, None)."""
        intrinsic_stack = getattr(self._rl_agent, "intrinsic_stack", None)
        if intrinsic_stack is None or intrinsic_stack.ngu_coeff <= 0:
            return None, None
        d = intrinsic_stack.ngu_feature_dim
        M = intrinsic_stack.max_seq_len
        mem = torch.zeros(batch_size, M, d, device=device)
        step_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        return mem, step_idx

    def init_hidden_state(self, batch_size: int, device) -> _BilevelHiddenState:
        ngu_mem, ngu_step_idx = self._init_ngu_state(batch_size, device)
        return _BilevelHiddenState(
            rl_hidden=self._rl_agent.init_hidden_state(batch_size, device),
            dyn_hidden=self._dyn.init_hidden_state(batch_size, device),
            e3b_sigma_inv=self._init_e3b_sigma_inv(batch_size, device),
            ngu_mem=ngu_mem,
            ngu_step_idx=ngu_step_idx,
        )

    def reset_hidden_state(self, hidden_state, dones):
        if hidden_state is None:
            return None
        if not isinstance(hidden_state, _BilevelHiddenState):
            return self._rl_agent.reset_hidden_state(hidden_state, dones)

        prev_rldp = hidden_state.prev_rldp
        prev_action = hidden_state.prev_action
        prev_valid = hidden_state.prev_valid
        prev_online = hidden_state.prev_online
        if prev_valid is not None:
            done_mask = torch.from_numpy(dones).bool().to(prev_valid.device)
            if prev_rldp is not None:
                prev_rldp = prev_rldp.clone()
                prev_rldp[done_mask] = 0.0
            if prev_action is not None:
                prev_action = prev_action.clone()
                prev_action[done_mask] = 0.0
            if prev_online is not None:
                prev_online = prev_online.clone()
                prev_online[done_mask] = 0.0
            prev_valid = prev_valid.clone()
            prev_valid[done_mask] = False

        e3b_sigma_inv = hidden_state.e3b_sigma_inv
        if e3b_sigma_inv is not None:
            done_mask = torch.from_numpy(dones).bool().to(e3b_sigma_inv.device)
            if done_mask.any():
                intrinsic_stack = self._rl_agent.intrinsic_stack
                d = intrinsic_stack.e3b_feature_dim
                lam = intrinsic_stack.e3b_lambda
                eye = (
                    torch.eye(
                        d,
                        device=e3b_sigma_inv.device,
                        dtype=e3b_sigma_inv.dtype,
                    )
                    / lam
                )
                e3b_sigma_inv = e3b_sigma_inv.clone()
                e3b_sigma_inv[done_mask] = eye

        ngu_mem = hidden_state.ngu_mem
        ngu_step_idx = hidden_state.ngu_step_idx
        if ngu_mem is not None and ngu_step_idx is not None:
            done_mask = torch.from_numpy(dones).bool().to(ngu_mem.device)
            if done_mask.any():
                ngu_mem = ngu_mem.clone()
                ngu_mem[done_mask] = 0.0
                ngu_step_idx = ngu_step_idx.clone()
                ngu_step_idx[done_mask] = 0

        prev_episode_id = hidden_state.prev_episode_id
        if prev_episode_id is not None and prev_valid is not None:
            prev_episode_id = prev_episode_id.clone()
            prev_episode_id[done_mask] = -1

        return _BilevelHiddenState(
            rl_hidden=self._rl_agent.reset_hidden_state(hidden_state.rl_hidden, dones),
            dyn_hidden=self._dyn.reset_hidden_state(hidden_state.dyn_hidden, dones),
            prev_rldp=prev_rldp,
            prev_action=prev_action,
            prev_valid=prev_valid,
            e3b_sigma_inv=e3b_sigma_inv,
            ngu_mem=ngu_mem,
            ngu_step_idx=ngu_step_idx,
            prev_episode_id=prev_episode_id,
            prev_online=prev_online,
        )


class BilevelAgent(nn.Module):
    """DynamicsLearner + RLAgent wrapper.

    The DynamicsLearner processes raw observations, augments the batch with
    detached 'rldp' features, and the RLAgent operates on the augmented batch.
    When using a NoOpDynamicsLearner, the batch passes through unchanged.
    """

    def __init__(
        self,
        dynamics_learner: DynamicsLearner,
        rl_agent: BaseAgent,
    ):
        super().__init__()
        assert isinstance(rl_agent, ZeroShotAgent), (
            f"BilevelAgent only supports ZeroShotAgent subclasses, got "
            f"{type(rl_agent).__name__}. Use standard AMAGO for meta-RL baselines."
        )
        self.dynamics_learner = dynamics_learner
        self.rl_agent = rl_agent

        self.update_info = {}
        self.pad_val = MAGIC_PAD_VAL
        self.discrete = rl_agent.discrete
        self.multibinary = rl_agent.multibinary
        self.action_dim = rl_agent.action_dim
        self.obs_space = rl_agent.obs_space
        self.rl2_space = rl_agent.rl2_space
        self.action_space = rl_agent.action_space
        self.max_seq_len = rl_agent.max_seq_len

        self._traj_encoder_proxy = _BilevelTrajEncoderProxy(
            rl_agent=rl_agent,
            dyn_traj_encoder=dynamics_learner.traj_encoder,
        )
        self._needs_rldp_error = "rldp_error" in rl_agent.obs_space.spaces

        # Running z-score of obs["observed"]. Published to batch/obs dicts as
        # "observed_normalized" so downstream consumers (DynamicsLearner's
        # diagnostic probes; IntrinsicCriticStack RND when configured with
        # rnd_obs_key="observed_normalized") read a single source of truth via
        # the batch/obs dict — mirroring how "rldp" / "rldp_error" flow.
        obs_dim = self.obs_space["observed"].shape[-1]
        self.obs_norm = InputNorm(obs_dim)

    def on_checkpoint_loaded(self, is_resume: bool = False):
        if hasattr(self.rl_agent, "on_checkpoint_loaded"):
            self.rl_agent.on_checkpoint_loaded(is_resume=is_resume)

    @property
    def traj_encoder(self):
        """Proxy that manages hidden state for both RL and dynamics encoders."""
        return self._traj_encoder_proxy

    def get_actions(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state=None,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, Any]:
        if isinstance(hidden_state, _BilevelHiddenState):
            dyn_hidden = hidden_state.dyn_hidden
            rl_hidden = hidden_state.rl_hidden
        else:
            dyn_hidden = None
            rl_hidden = hidden_state

        obs["observed_normalized"] = self.obs_norm(obs["observed"]).detach()

        # Publish previous extrinsic reward as a dedicated obs key so the
        # ExtrinsicTstepEncoder can consume it without breaking the
        # ``hide_rl2s=True`` invariant.  Mirrors the train-time transcription
        # in BilevelAgent.forward (rl2s[..., 0] is the canonical previous
        # extrinsic reward in both train and rollout paths).
        obs["ext_reward"] = sanitize_published("ext_reward", rl2s[..., 0:1])

        rldp, new_dyn_hidden = self.dynamics_learner.encode(
            obs, rl2s, time_idxs, dyn_hidden
        )
        obs["rldp"] = sanitize_published("rldp", rldp)

        cached_idm = self.dynamics_learner.traj_encoder._cached_idm
        if cached_idm is not None:
            obs["idm"] = sanitize_published("idm", cached_idm)
        else:
            obs["idm"] = torch.zeros(*rldp.shape[:-1], 1, device=rldp.device)

        cached_tstep_emb = self.dynamics_learner._cached_tstep_emb
        if cached_tstep_emb is not None:
            obs["dyn_tstep_emb"] = sanitize_published("dyn_tstep_emb", cached_tstep_emb)
        else:
            obs["dyn_tstep_emb"] = torch.zeros(*rldp.shape[:-1], 1, device=rldp.device)

        prev_valid = (
            hidden_state.prev_valid
            if isinstance(hidden_state, _BilevelHiddenState)
            else None
        )
        if prev_valid is not None:
            prev_valid_mask = prev_valid
        else:
            prev_valid_mask = torch.zeros(
                *rldp.shape[:-1], 1, device=rldp.device, dtype=torch.bool
            )

        if self._needs_rldp_error:
            if prev_valid is not None and prev_valid.any():
                raw_error = self.dynamics_learner.compute_prediction_error(
                    hidden_state.prev_rldp,
                    hidden_state.prev_action,
                    obs,
                    rl2s,
                    prev_online=hidden_state.prev_online,
                )
                obs["rldp_error"] = sanitize_published(
                    "rldp_error", raw_error * prev_valid.float()
                )
            else:
                obs["rldp_error"] = torch.zeros(*rldp.shape[:-1], 1, device=rldp.device)

        # Composite per-step intrinsic reward consumed by the policy trajectory
        # encoder.  Matches the training-time signal from
        # IntrinsicCriticStack.compute_intrinsic_reward (same RND + rldp_error
        # formula, same RMS-normalized scale, zero on the first step of a
        # trial).  Uses frozen RMS stats at rollout (update only during
        # training).  Fallback to obs["rldp_error"] when no intrinsic stack is
        # attached so traj_encoder_pi still sees a well-defined side channel.
        intrinsic_stack = getattr(self.rl_agent, "intrinsic_stack", None)
        new_e3b_sigma_inv: Optional[torch.Tensor] = None
        new_ngu_mem: Optional[torch.Tensor] = None
        new_ngu_step_idx: Optional[torch.Tensor] = None
        if intrinsic_stack is not None:
            prev_sigma = (
                hidden_state.e3b_sigma_inv
                if isinstance(hidden_state, _BilevelHiddenState)
                else None
            )
            prev_ngu_mem = (
                hidden_state.ngu_mem
                if isinstance(hidden_state, _BilevelHiddenState)
                else None
            )
            prev_ngu_step_idx = (
                hidden_state.ngu_step_idx
                if isinstance(hidden_state, _BilevelHiddenState)
                else None
            )
            # Build episodic reset mask: 0 at episode-reset boundaries,
            # 1 elsewhere.  Reused for rldp_error, E3B, and NGU components
            # (reset transitions are uncontrollable env teleports — their
            # "novelty" is not exploration so we zero those components).
            episodic_reset_mask = None
            prev_ep = (
                hidden_state.prev_episode_id
                if isinstance(hidden_state, _BilevelHiddenState)
                else None
            )
            if prev_ep is not None:
                cur_ep = obs["episode_id"]
                cur_ep_flat = cur_ep.squeeze(-1) if cur_ep.ndim > 1 else cur_ep
                ep_changed = cur_ep_flat != prev_ep  # (B,)
                episodic_reset_mask = (~ep_changed).float().view(-1, 1, 1)

            (
                ir_raw,
                new_e3b_sigma_inv,
                new_ngu_mem,
                new_ngu_step_idx,
            ) = intrinsic_stack.compute_intrinsic_reward_step(
                obs,
                valid_mask=prev_valid_mask,
                e3b_sigma_inv=prev_sigma,
                episodic_reset_mask=episodic_reset_mask,
                ngu_mem=prev_ngu_mem,
                ngu_step_idx=prev_ngu_step_idx,
            )
            obs["intrinsic_reward"] = sanitize_published("intrinsic_reward", ir_raw)
        else:
            obs["intrinsic_reward"] = obs["rldp_error"]

        actions, new_rl_hidden = self.rl_agent.get_actions(
            obs=obs,
            rl2s=rl2s,
            time_idxs=time_idxs,
            hidden_state=rl_hidden,
            sample=sample,
        )

        _prev_valid_next = torch.ones(
            *rldp.shape[:-1], 1, device=rldp.device, dtype=torch.bool
        )
        _cur_ep = obs["episode_id"]
        _cur_ep = _cur_ep.squeeze(-1) if _cur_ep.ndim > 1 else _cur_ep
        _cur_ep = _cur_ep.detach()
        _prev_online_next = None
        if self.dynamics_learner.skip_prev_target:
            cached = self.dynamics_learner._cached_tstep_emb
            if cached is not None:
                _prev_online_next = cached.detach()
        if self._needs_rldp_error:
            new_hidden = _BilevelHiddenState(
                rl_hidden=new_rl_hidden,
                dyn_hidden=new_dyn_hidden,
                prev_rldp=rldp.detach(),
                prev_action=actions.detach(),
                prev_valid=_prev_valid_next,
                e3b_sigma_inv=new_e3b_sigma_inv,
                ngu_mem=new_ngu_mem,
                ngu_step_idx=new_ngu_step_idx,
                prev_episode_id=_cur_ep,
                prev_online=_prev_online_next,
            )
        else:
            new_hidden = _BilevelHiddenState(
                rl_hidden=new_rl_hidden,
                dyn_hidden=new_dyn_hidden,
                prev_valid=_prev_valid_next,
                e3b_sigma_inv=new_e3b_sigma_inv,
                ngu_mem=new_ngu_mem,
                ngu_step_idx=new_ngu_step_idx,
                prev_episode_id=_cur_ep,
                prev_online=_prev_online_next,
            )
        return actions, new_hidden

    def forward(self, batch: Batch, log_step: bool) -> torch.Tensor:
        self.update_info = {}
        if self.training:
            with torch.no_grad():
                self.obs_norm.update_stats(batch.obs["observed"])
        batch.obs["observed_normalized"] = self.obs_norm(batch.obs["observed"]).detach()

        batch_seq_len = batch.rl2s.shape[1]
        dynamics_loss = self.dynamics_learner(batch, log_step)
        self.update_info.update(self.dynamics_learner.update_info)

        # Intrinsic reward at the ORIGINAL (pre-slice) sequence length so
        # stateful intrinsic components (E3B Σ, NGU memory) see a full
        # meta-trial.  Writes ``batch.obs["intrinsic_reward"]`` in place
        # and returns an optional differentiable predictor loss (RND)
        # added to the total below.
        intrinsic_reward_loss = self.rl_agent.compute_intrinsic_reward(batch, log_step)

        # Publish previous extrinsic reward as a dedicated obs key so the
        # ExtrinsicTstepEncoder can consume it without breaking the
        # ``hide_rl2s=True`` invariant on TDJepa's encoders.  AMAGO's
        # Trajectory.__post_init__ constructs ``rl2s[..., 0]`` as the
        # previous extrinsic reward (zero-padded at t=0), so this is 1:1
        # with the rollout-time RL2 wrapper convention.
        batch.obs["ext_reward"] = batch.rl2s[..., 0:1].detach()

        # Optional slice from max_seq_len down to rl_agent.ac_seq_len.  No-op
        # unless ac_seq_len has been set on the rl_agent (TDJepa via
        # gin).  Gathers all batch.obs keys plus rl2s/rews/dones/actions/
        # time_idxs so downstream §1 backbones see a coherent sliced batch.
        batch = self.rl_agent.slice_batch(batch)
        sliced_seq_len = batch.rl2s.shape[1]

        rl_loss = self.rl_agent(batch, log_step=log_step)
        self.update_info.update(self.rl_agent.update_info)

        if log_step:
            # Per-step visibility into the three sequence lengths used across
            # the pipeline: full batch (dynamics_learner + compute_intrinsic_
            # reward), post-slice (actor + intrinsic critic in TDJepa §1/§2),
            # and TDJepa §3 MLP window ("Seq/MLP Window Length", logged by
            # TDJepa.compute_losses itself).
            self.update_info["Seq/Batch Length"] = batch_seq_len
            self.update_info["Seq/After Slice Length"] = sliced_seq_len
            if "observed" in batch.obs:
                self.update_info["Obs/observed_dim"] = batch.obs["observed"].shape[-1]

        total = dynamics_loss + rl_loss
        if intrinsic_reward_loss is not None:
            total = total + intrinsic_reward_loss
        return total

    @property
    def trainable_params(self):
        import itertools

        return itertools.chain(
            self.dynamics_learner.trainable_params,
            self.rl_agent.trainable_params,
        )

    def clip_param_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """Compose dynamics + RL-agent clip groups for grouped grad clipping.

        Returns four disjoint param subsets keyed by clip-group name. Used by
        ``BilevelExperiment.train_step`` to norm-clip dynamics / amago / rnd
        and value-clip tdjepa_core (the SF backbone).
        """
        groups = {"dynamics": list(self.dynamics_learner.trainable_params)}
        if hasattr(self.rl_agent, "clip_param_groups"):
            groups.update(self.rl_agent.clip_param_groups())
        else:
            groups.update(
                {
                    "amago": [],
                    "tdjepa_core": list(self.rl_agent.trainable_params),
                    "rnd": [],
                }
            )
        return groups

    def hard_sync_targets(self):
        self.dynamics_learner.hard_sync_targets()
        self.rl_agent.hard_sync_targets()

    def soft_sync_targets(self):
        self.dynamics_learner.soft_sync_targets()
        self.rl_agent.soft_sync_targets()

    def get_grad_norms(self) -> dict[str, float]:
        norms = {}
        norms.update(self.dynamics_learner.get_grad_norms())
        norms.update(self.rl_agent.get_grad_norms())
        return norms


# ──────────────────────────────────────────────────────────────────────────────
# Experiment
# ──────────────────────────────────────────────────────────────────────────────


@gin.configurable
class BilevelExperiment(Experiment):
    """Experiment for BilevelAgent (DynamicsLearner + RLAgent).

    Evaluation (``evaluate_val`` / ``evaluate_test``) runs up to three
    *phases* (α=β=0; ``val_alpha_schedule``; α=1 with ``val_beta_schedule``),
    each made of *rounds*: one ``(reset inference z, run one trial, record
    returns)`` per parallel actor.  See :meth:`_run_eval_phase`.
    """

    def __init__(
        self,
        rl_agent_type: type = gin.REQUIRED,
        dynamics_learner_type: type = gin.REQUIRED,
        num_collection_batches: Optional[int] = None,
        steps_per_round: Optional[int] = None,
        resample_alpha_per_episode: bool = True,
        baseline_preset: Optional[str] = None,
        gin_ladder: Optional[list] = None,
        clip_norm: float = 2.0,
        **kwargs,
    ):
        self.rl_agent_type = rl_agent_type
        self.dynamics_learner_type = dynamics_learner_type
        self._jsonl_path: Optional[str] = None
        self._jsonl_file = None
        self.num_collection_batches = num_collection_batches
        self.steps_per_round = steps_per_round
        self.resample_alpha_per_episode = resample_alpha_per_episode
        self._resample_alpha_per_episode_active = False
        self._prev_ep_id: Optional[torch.Tensor] = None
        # Global L2 norm clip for the dynamics, AMAGO, and RND clip groups.
        # The TDJepa-core group is value-clipped via the parent ``grad_clip``
        # knob instead. See ``train_step``.
        self.clip_norm = clip_norm
        # Baseline-preset bookkeeping for the startup banner, populated by
        # the ``train_*.py`` launchers; None for non-launcher callers.
        self._baseline_preset: Optional[str] = baseline_preset
        self._gin_ladder: Optional[list] = list(gin_ladder) if gin_ladder else None
        super().__init__(**kwargs)

    # ── Local logging ──────────────────────────────────────────────────────

    def init_logger(self) -> None:
        super().init_logger()
        if self.accelerator.is_main_process:
            run_dir = os.path.dirname(self.ckpt_dir)
            self._jsonl_path = os.path.join(run_dir, "history.jsonl")
            self._jsonl_file = open(self._jsonl_path, "a")

    def log(self, metrics_dict, key: str) -> None:
        super().log(metrics_dict, key)
        if self._jsonl_file is not None:
            import json as _json

            def _scalar(v):
                if isinstance(v, torch.Tensor) and v.ndim == 0:
                    return v.detach().cpu().float().item()
                if isinstance(v, (np.integer,)):
                    return int(v)
                if isinstance(v, (np.floating,)):
                    return float(v)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return v
                return None

            row = {}
            for k, v in metrics_dict.items():
                s = _scalar(v)
                if s is not None:
                    row[f"{key}/{k}"] = s
            for k, v in self.x_axis_metrics().items():
                s = _scalar(v)
                if s is not None:
                    row[k] = s
            self._jsonl_file.write(_json.dumps(row) + "\n")
            self._jsonl_file.flush()

    # ── Dataset / Dataloader ──────────────────────────────────────────────

    def init_dloaders(self) -> DataLoader:
        """Create dataloaders."""
        train_dloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.dloader_workers,
            collate_fn=RLData_pad_collate,
            pin_memory=True,
        )
        self.train_dloader = self.accelerator.prepare(train_dloader)
        return self.train_dloader

    # ── Optimization ──────────────────────────────────────────────────────

    def init_optimizer(self, policy) -> torch.optim.Optimizer:
        """Filter out frozen params so AdamW doesn't allocate state for them."""
        params = [p for p in policy.trainable_params if p.requires_grad]
        return torch.optim.AdamW(
            params, lr=self.learning_rate, weight_decay=self.l2_coeff
        )

    def train_step(self, batch, log_step: bool):
        """Grouped grad clipping: norm-clip three subsystems, value-clip one.

        The four groups from ``policy.clip_param_groups()`` (dynamics / amago /
        tdjepa_core / rnd) are disjoint gradient subgraphs by construction
        (``detach_actor=True`` cuts phi at the actor input;
        ``sanitize_published`` cuts dynamics outputs; the SF critic loss uses
        buffer actions; RND's input and target are detached), so per-group
        clipping has no cross-talk.

          * ``dynamics`` / ``amago`` / ``rnd`` → ``clip_grad_norm_(clip_norm)``
          * ``tdjepa_core``                    → ``clip_grad_value_(grad_clip)``
            (a stability backstop for the SF backbone).

        ``clip_norm <= 0`` / ``grad_clip <= 0`` disable the respective clips.
        Pre-clip L2 norms are logged under ``GradClip/Pre-Clip Norm/{group}``.
        """
        policy = self.policy
        # Always use the grouped-clip path rather than the parent's
        # train_step; empty groups (e.g. dynamics under NoOpDynamicsLearner)
        # are skipped below.
        with self.accelerator.accumulate(self.policy_aclr):
            self.optimizer.zero_grad()
            loss = self.policy_aclr(batch, log_step=log_step)
            l = {"Loss": loss, **self.policy.update_info}
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                groups = policy.clip_param_groups()

                # clip_grad_norm_ returns the pre-clip L2 norm, so the
                # diagnostic is free.
                norm_clip_active = self.clip_norm is not None and self.clip_norm > 0
                for name in ("dynamics", "amago", "rnd"):
                    params_with_grad = [
                        p
                        for p in groups.get(name, [])
                        if p.requires_grad and p.grad is not None
                    ]
                    if not params_with_grad:
                        continue
                    if norm_clip_active:
                        pre_norm = torch.nn.utils.clip_grad_norm_(
                            params_with_grad, max_norm=self.clip_norm
                        )
                        if log_step:
                            l[f"GradClip/Pre-Clip Norm/{name}"] = pre_norm.item()
                    elif log_step:
                        # Norm clip disabled: still log the natural L2 norm so
                        # the grad-health diagnostic stays comparable.
                        norms = torch._foreach_norm(
                            [p.grad.detach() for p in params_with_grad]
                        )
                        l[f"GradClip/Pre-Clip Norm/{name}"] = torch.norm(
                            torch.stack(norms)
                        ).item()

                # Per-element value clip on TDJepa-core only. ``grad_clip <= 0``
                # disables clipping entirely; without the guard,
                # ``clip_grad_value_(p, 0)`` would clamp every gradient to
                # zero and silently break training.
                core_with_grad = [
                    p
                    for p in groups.get("tdjepa_core", [])
                    if p.requires_grad and p.grad is not None
                ]
                if core_with_grad:
                    if log_step:
                        # Manual pre-clip norm via foreach_norm — value clip
                        # has no useful return value to recycle.
                        norms = torch._foreach_norm(
                            [p.grad.detach() for p in core_with_grad]
                        )
                        l["GradClip/Pre-Clip Norm/tdjepa_core"] = torch.norm(
                            torch.stack(norms)
                        ).item()
                    if self.grad_clip is not None and self.grad_clip > 0:
                        torch.nn.utils.clip_grad_value_(core_with_grad, self.grad_clip)

                self.policy.soft_sync_targets()
                self.grad_update_counter += 1
                if log_step:
                    l.update(
                        {
                            "Learning Rate": self.lr_schedule.get_last_lr()[0],
                            **self.policy.get_grad_norms(),
                        }
                    )
            self.optimizer.step()
            self.lr_schedule.step()
        return l

    # ── Model ─────────────────────────────────────────────────────────────

    def init_model(self) -> None:
        """Build the BilevelAgent: DynamicsLearner + RLAgent."""
        obs_space = self.rl2_space["obs"]
        rl2_space = self.rl2_space["rl2"]
        action_space = self.train_envs.single_action_space
        max_seq_len = self.max_seq_len

        policy = self._init_dynamics_model(
            obs_space, rl2_space, action_space, max_seq_len
        )

        optimizer = self.init_optimizer(policy)
        lr_schedule = utils.get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=self.lr_warmup_steps
        )
        self.policy_aclr, self.optimizer, self.lr_schedule = self.accelerator.prepare(
            policy, optimizer, lr_schedule
        )
        self.accelerator.register_for_checkpointing(self.lr_schedule)
        self.grad_update_counter = 0

    def _init_dynamics_model(self, obs_space, rl2_space, action_space, max_seq_len):
        """Build DynamicsLearner + RLAgent wrapped in BilevelAgent.

        Uses gin scopes to isolate encoder configs:
          - ``dynamics/`` for the DynamicsLearner and its sub-encoders
          - ``agent/`` for the RL agent and its sub-encoders
        """
        with gin.config_scope("dynamics"):
            dynamics_learner = self.dynamics_learner_type(
                obs_space=obs_space,
                rl2_space=rl2_space,
                action_space=action_space,
                max_seq_len=max_seq_len,
            )

        # Augment obs space with the keys BilevelAgent publishes (rldp,
        # rldp_error, idm, dyn_tstep_emb, intrinsic_reward, ext_reward, alpha,
        # observed_normalized) so the RL agent's encoders /
        # IntrinsicCriticStack know their input dimensions at construction.
        # - rldp_error is inert unless the agent's specify_obs_keys includes it.
        # - intrinsic_reward is the composite per-step novelty signal (see
        #   IntrinsicCriticStack.compute_intrinsic_reward), consumed by
        #   TDJepa's traj_encoder_pi as a policy side channel.
        # - observed_normalized is the running-z-scored obs["observed"].
        rldp_dim = dynamics_learner.rldp_dim
        idm_dim = dynamics_learner.idm_dim
        dyn_tstep_dim = dynamics_learner.dyn_tstep_dim
        obs_box = obs_space.spaces["observed"]
        rl_obs_space = gym.spaces.Dict(
            {
                **obs_space.spaces,
                "rldp": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(rldp_dim,), dtype=np.float32
                ),
                "rldp_error": gym.spaces.Box(
                    low=0.0, high=2.0, shape=(1,), dtype=np.float32
                ),
                "idm": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(max(idm_dim, 1),),
                    dtype=np.float32,
                ),
                "dyn_tstep_emb": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(max(dyn_tstep_dim, 1),),
                    dtype=np.float32,
                ),
                "intrinsic_reward": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "ext_reward": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "alpha": gym.spaces.Box(
                    low=0.0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "observed_normalized": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=obs_box.shape,
                    dtype=obs_box.dtype,
                ),
            }
        )

        with gin.config_scope("agent"):
            rl_agent = self.rl_agent_type(
                obs_space=rl_obs_space,
                rl2_space=rl2_space,
                action_space=action_space,
                max_seq_len=max_seq_len,
            )
        if hasattr(rl_agent, "tstep_encoder"):
            rl_agent.tstep_encoder.hide_rl2s = True

        return BilevelAgent(dynamics_learner=dynamics_learner, rl_agent=rl_agent)

    # ── Summary ───────────────────────────────────────────────────────────

    @staticmethod
    def _describe_traj_encoder(enc) -> str:
        """Format a TrajEncoder as a one-liner with dims + attention info.

        Recognises ``IdentityTrajEncoder``, ``FFTrajEncoder``, ``TformerTrajEncoder``
        and the ``DynamicsEncoder`` wrapper used by the dynamics learner.
        """
        if enc is None:
            return "None"
        cls = enc.__class__.__name__
        # Unwrap the dynamics encoder wrapper (we want the inner trunk's specs).
        inner = getattr(enc, "dynamics_encoder", None)
        if inner is not None:
            wrapper_dim = getattr(enc, "emb_dim", None)
            inner_str = BilevelExperiment._describe_traj_encoder(inner)
            extras = [f"out_dim={wrapper_dim}"]
            if getattr(enc, "sphere_normalize", False):
                extras.append("sphere_norm")
            return f"{cls}({inner_str}, {', '.join(extras)})"
        # Tformer
        if hasattr(enc, "tformer"):
            d_model = getattr(enc, "d_model", "?")
            n_heads = getattr(enc, "n_heads", "?")
            n_layers = getattr(enc, "n_layers", "?")
            max_seq_len = getattr(enc, "max_seq_len", "?")
            # Probe the first attention block for type + window + pos_emb.
            attn_str = "?"
            pos_emb = "fixed"
            for _, mod in enc.named_modules():
                mod_cls = type(mod).__name__
                if (
                    mod_cls in ("FlashAttention", "VanillaAttention")
                    and attn_str == "?"
                ):
                    if hasattr(mod, "window_size") and mod.window_size != (-1, -1):
                        attn_str = f"{mod_cls}(window={tuple(mod.window_size)})"
                    else:
                        attn_str = mod_cls
                if "RotaryPosEmb" in mod_cls:
                    pos_emb = "rope"
            return (
                f"{cls}(d_model={d_model}, heads={n_heads}, layers={n_layers}, "
                f"max_seq_len={max_seq_len}, pos_emb={pos_emb}, attn={attn_str})"
            )
        # FF / Identity (single line of context)
        max_seq_len = getattr(enc, "max_seq_len", "?")
        emb_dim = getattr(enc, "emb_dim", None)
        bits = [f"max_seq_len={max_seq_len}"]
        if emb_dim is not None:
            bits.append(f"emb_dim={emb_dim}")
        return f"{cls}({', '.join(bits)})"

    @staticmethod
    def _describe_mlp(mlp) -> str:
        """Format an ``amago.nets.ff.MLP`` (or a JEPA-style stack) one-liner.

        Reports d_inp -> d_hidden x N -> d_out so the actor / phi / psi /
        PolicyTstepEncoder MLP shapes are visible at a glance.
        """
        if mlp is None:
            return "None"
        # amago.nets.ff.MLP layout
        in_layer = getattr(mlp, "in_layer", None)
        out_layer = getattr(mlp, "out_layer", None)
        layers = getattr(mlp, "layers", None)
        if in_layer is not None and out_layer is not None and layers is not None:
            d_inp = in_layer.in_features
            d_hidden = in_layer.out_features
            d_output = out_layer.out_features
            n_layers = len(layers) + 1
            return f"MLP({d_inp} -> {d_hidden} x {n_layers} -> {d_output})"
        # JepaEncoder layout: inp + (n_layers Linear blocks) + final Linear.
        inp = getattr(mlp, "inp", None)
        net = getattr(mlp, "net", None)
        if inp is not None and net is not None and not hasattr(net, "inp_layer"):
            d_inp = inp.in_features
            d_hidden = inp.out_features
            d_output = getattr(mlp, "emb_dim", None)
            n_hidden_blocks = sum(
                1
                for m in net
                if isinstance(m, nn.Linear) and m.out_features == d_hidden
            )
            return (
                f"JepaEncoder({d_inp} -> {d_hidden} x {n_hidden_blocks + 1}"
                f" -> {d_output})"
            )
        return mlp.__class__.__name__

    @staticmethod
    def _describe_jepa_predictor(pred) -> str:
        """Format a JepaPredictor (ensembled EinMix MLP) one-liner.

        JepaPredictor wraps an ``_EinMixEnsemble`` of ``ensemble_size`` parallel
        MLPs (Mix(d_in, d_hidden) -> n_layers-1 Mix(d_hidden, d_hidden) ->
        Mix(d_hidden, out_dim)). We pull the dims from the einops Mix weight
        shape ``[ensemble, d_in, d_out]``.
        """
        if pred is None:
            return "None"
        net = getattr(pred, "net", None)
        if net is None:
            return pred.__class__.__name__
        inp_layer = getattr(net, "inp_layer", None)
        core_layers = getattr(net, "core_layers", None)
        out_layer = getattr(net, "output_layer", None)
        if inp_layer is None or out_layer is None or core_layers is None:
            return pred.__class__.__name__
        # einops EinMix stores ``weight`` of shape (c, d_in, d_out).
        ensemble, d_in, d_hidden = tuple(inp_layer.weight.shape)
        d_out = int(out_layer.weight.shape[-1])
        n_layers = len(core_layers) + 1  # inp_layer counts as the first layer
        return f"EinMix x{ensemble}({d_in} -> {d_hidden} x {n_layers} -> {d_out})"

    @staticmethod
    def _describe_tstep_encoder(enc, label: str | None = None) -> str:
        """Format a TstepEncoder one-liner with emb_dim + obs_keys + flags."""
        if enc is None:
            return "None"
        cls = enc.__class__.__name__
        bits = []
        emb_dim = getattr(enc, "emb_dim", None)
        if emb_dim is not None:
            bits.append(f"emb_dim={emb_dim}")
        obs_keys = (
            getattr(enc, "obs_keys", None)
            or getattr(enc, "_obs_keys", None)
            or getattr(enc, "REQUIRED_OBS_KEYS", None)
        )
        if obs_keys is not None:
            bits.append(f"obs_keys={list(obs_keys)}")
        for flag in ("normalize_inputs", "hide_rl2s", "hide_rewards"):
            v = getattr(enc, flag, None)
            if v is not None:
                bits.append(f"{flag}={v}")
        prefix = f"{label}: " if label else ""
        return f"{prefix}{cls}({', '.join(bits)})"

    def _seq_lengths_summary(self, rl_agent, dyn) -> str:
        """Show train-time effective L for each module, plus constructor cap.

        ``BilevelAgent.forward`` runs:
          1. dynamics_learner(batch)            at L = max_seq_len
          2. rl_agent.compute_intrinsic_reward  at L = max_seq_len
          3. rl_agent.slice_batch -> ac_seq_len if set
          4. rl_agent(batch)                    at L = ac_seq_len (post-slice)

        Every traj encoder is sized for ``max_seq_len``, but the RL agent's
        traj encoders only see ``ac_seq_len`` tokens at training time.
        """
        max_seq_len = self.max_seq_len
        ac_seq_len = rl_agent.ac_seq_len
        mlp_seq_len = rl_agent.mlp_seq_len

        def _eff(post_slice: bool) -> str:
            if post_slice and ac_seq_len < max_seq_len:
                return f"L={ac_seq_len} (sliced from {max_seq_len})"
            return f"L={max_seq_len}"

        lines = [
            f"max_seq_len (env / rollout window):  {max_seq_len}",
            f"ac_seq_len  (post-slice for RL):     {ac_seq_len}",
            f"mlp_seq_len (phi/psi predictor):     {mlp_seq_len}",
            "Per-module train-time forward length:",
        ]

        def _cap(enc) -> int:
            return getattr(enc, "max_seq_len", max_seq_len)

        if not isinstance(dyn, NoOpDynamicsLearner):
            dyn_enc = getattr(dyn.traj_encoder, "dynamics_encoder", dyn.traj_encoder)
            lines.append(
                f"  DynamicsLearner traj enc:        {_eff(False)} | cap={_cap(dyn_enc)}"
            )
        ic = getattr(rl_agent, "intrinsic_stack", None)
        if ic is not None:
            ic_traj = getattr(ic, "traj_encoder", None)
            if ic_traj is not None:
                lines.append(
                    "  Intrinsic-reward forward (full): "
                    f"{_eff(False)} | cap={_cap(ic_traj)}"
                )
                lines.append(
                    "  Intrinsic-critic TD forward:     "
                    f"{_eff(True)} | cap={_cap(ic_traj)}"
                )
        sf_traj = getattr(rl_agent, "traj_encoder", None)
        if sf_traj is not None:
            lines.append(
                f"  TDJepa phi/psi traj enc:         {_eff(True)} | cap={_cap(sf_traj)}"
            )
        pi_traj = getattr(rl_agent, "traj_encoder_pi", None)
        if pi_traj is not None:
            lines.append(
                f"  Policy traj enc:                 {_eff(True)} | cap={_cap(pi_traj)}"
            )
        if mlp_seq_len is not None:
            lines.append(
                f"  TDJepa phi/psi predictor MLPs:   L={mlp_seq_len} (windowed inside §3)"
            )
            lines.append(
                f"  TDJepa V_td / actor critic loss: L={mlp_seq_len} (same §3 window)"
            )
        return "\n    ".join(lines)

    def _dynamics_summary(self, dyn) -> str:
        if isinstance(dyn, NoOpDynamicsLearner):
            return "NoOpDynamicsLearner (disabled)"
        parts = []
        flags = []
        if getattr(dyn, "skip_prev_target", False):
            flags.append("skip_prev_target=True")
        ldp_target = getattr(dyn, "ldp_target", None)
        if ldp_target is not None:
            flags.append(f"ldp_target={ldp_target}")
        probes_mode = "detached" if getattr(dyn, "detach_probes", True) else "attached"
        flags.append(f"probes={probes_mode}")
        ema_tau = getattr(dyn, "ldp_ema_tau", None)
        if ema_tau is not None:
            flags.append(f"ema_tau={ema_tau}")
        parts.append(f"Class: {dyn.__class__.__name__} | {' | '.join(flags)}")

        # Identify the dynamics regime cell: LDP (latent prediction) vs
        # FDM (forward dynamics modeling, probes attached, LDP off).
        ldp_coeff = getattr(dyn, "ldp_coeff", 0.0) or 0.0
        ortho_coeff = getattr(dyn, "ldp_ortho_coeff", 0.0) or 0.0
        if ldp_coeff > 0:
            regime = (
                f"Dynamics: LDP (probes detached, ortho={ortho_coeff})"
                if probes_mode == "detached"
                else f"Dynamics: LDP+probes (probes attached, ortho={ortho_coeff})"
            )
        else:
            regime = (
                "Dynamics: FDM (probes attached, LDP off)"
                if probes_mode == "attached"
                else "Dynamics: IDM-only (probes detached, LDP off)"
            )
        parts.append(regime)

        objectives = []
        if getattr(dyn, "idm_coeff", 0) > 0:
            objectives.append(f"IDM(coeff={dyn.idm_coeff})")
        if ldp_coeff > 0:
            objectives.append(f"LDP/{ldp_target or '?'}(coeff={dyn.ldp_coeff})")
        if ortho_coeff > 0:
            objectives.append(f"LDP-ortho(coeff={dyn.ldp_ortho_coeff})")
        objectives.append(f"obs probes ({probes_mode})")
        parts.append(f"Objectives: {' + '.join(objectives)}")

        parts.append(
            "  " + self._describe_tstep_encoder(dyn.tstep_encoder, label="Tstep enc")
        )
        parts.append(f"  Traj enc:   {self._describe_traj_encoder(dyn.traj_encoder)}")
        head_bits = [f"RLDP={dyn.rldp_dim}"]
        if getattr(dyn, "idm_coeff", 0) > 0 and dyn.idm_dim > 0:
            head_bits.append(f"IDM={dyn.idm_dim}")
        if hasattr(dyn, "dyn_tstep_dim"):
            head_bits.append(f"dyn_tstep={dyn.dyn_tstep_dim}")
        parts.append("  Heads:      " + ", ".join(head_bits))
        dyn_params = sum(p.numel() for p in dyn.parameters())
        parts.append(f"  Parameters: {dyn_params:,d}")
        return "\n    ".join(parts)

    @staticmethod
    def _describe_axis_dist(label: str, dist, schedule, n_slots) -> str:
        """Render one α or β axis as a single banner line (incl. val schedule)."""
        if dist is None:
            return f"{label}: <unset>"
        bits = [f"max_value={dist.max_value}"]
        for attr in ("p_zero", "p_one"):
            if hasattr(dist, attr):
                bits.append(f"{attr}={getattr(dist, attr)}")
        if n_slots is not None:
            bits.append(f"n={n_slots}")
        if schedule is not None:
            sched_name = schedule.__class__.__name__
            if sched_name == "EpisodeThresholdSchedule":
                hi = getattr(schedule, "_high", "?")
                lo = getattr(schedule, "_low", "?")
                thr = getattr(schedule, "_threshold", "?")
                bits.append(
                    f"val_sched={sched_name}({label.lower()}={hi}->{lo} @ ep {thr})"
                )
            elif sched_name == "ConstantSchedule":
                inner = getattr(schedule, "dist", None)
                inner_v = getattr(inner, "max_value", "?") if inner else "?"
                bits.append(f"val_sched={sched_name}({label.lower()}={inner_v})")
            else:
                bits.append(f"val_sched={sched_name}(max_value={schedule.max_value})")
        return f"{label}: {dist.__class__.__name__}({', '.join(bits)})"

    def _alpha_beta_summary(self, rl_agent) -> str:
        """Banner block for α-β conditioning + extrinsic-encoder branch.

        Always renders both axes (even degenerate constant-0 ones) so the
        user can verify which baseline preset is loaded just by reading
        the banner. Slot grid line shows the cross-product layout used by
        ``compute_losses`` and which corner cells are pinned.
        """
        from metabfm.rl.conditioning import fourier_encoded_dim

        alpha_dist = getattr(rl_agent, "alpha_dist", None)
        beta_dist = getattr(rl_agent, "beta_dist", None)
        n_alphas = getattr(rl_agent, "n_alphas", 1)
        n_betas = getattr(rl_agent, "n_betas", 1)
        val_alpha_sched = getattr(rl_agent, "val_alpha_schedule", None)
        val_beta_sched = getattr(rl_agent, "val_beta_schedule", None)
        n_freq = getattr(rl_agent, "fourier_n_freq", 16)
        per_scalar = fourier_encoded_dim(n_freq)
        ext_mult = getattr(rl_agent, "extrinsic_reward_multiplier", 1.0)
        ext_traj = getattr(rl_agent, "extrinsic_traj_encoder", None)

        parts = []
        parts.append(
            self._describe_axis_dist("Alpha", alpha_dist, val_alpha_sched, n_alphas)
        )
        parts.append(
            self._describe_axis_dist("Beta ", beta_dist, val_beta_sched, n_betas)
        )
        parts.append(
            f"Slot grid: {n_alphas} x {n_betas} = {n_alphas * n_betas} (alpha, beta) "
            f"combinations per row"
        )
        # Rollout-time AMAGO-faithfulness pin: deterministic fraction of
        # actors held at (α=1, β=1) every collection round. No-op unless
        # full αβ is active (n_alphas > 1 AND n_betas > 1).
        pin_frac = getattr(rl_agent, "pin_pure_extrinsic_fraction", 0.0)
        if pin_frac > 0.0 and n_alphas > 1 and n_betas > 1:
            parts.append(
                f"Rollout pin: pin_pure_extrinsic_fraction={pin_frac:.2f} "
                f"(deterministic AMAGO-faithful (α=1, β=1) corner)"
            )
        elif pin_frac > 0.0:
            parts.append(
                f"Rollout pin: pin_pure_extrinsic_fraction={pin_frac:.2f} "
                f"[INACTIVE: requires n_alphas > 1 AND n_betas > 1]"
            )
        # Pinned corners follow the same NA>=2 / NB>=2 gates used by
        # compute_losses.
        if n_alphas >= 2 and n_betas >= 2:
            corners = "(α=0, β=0), (α=max, β=0), (α=max, β=max)"
        elif n_alphas >= 2:
            corners = "(α=0, β=const), (α=max, β=const)"
        elif n_betas >= 2:
            corners = "(α=const, β=0), (α=const, β=max)"
        else:
            corners = f"(α={alpha_dist.max_value if alpha_dist else '?'}, "
            corners += f"β={beta_dist.max_value if beta_dist else '?'}) [degenerate, no pinning]"
        parts.append(f"Pinned corners: {corners}")
        parts.append(
            f"Fourier encoding: n_freq={n_freq} ({per_scalar} dims per scalar, "
            f"{2 * per_scalar} total for α+β; raw scalar appended as safety tail)"
        )
        if ext_traj is None:
            parts.append("Ext encoder: NOT CONSTRUCTED (β_dist.max_value == 0)")
        else:
            ext_desc = self._describe_traj_encoder(ext_traj)
            parts.append(f"Ext encoder: {ext_desc}")
        parts.append(f"Ext reward multiplier: {ext_mult}")
        return "\n    ".join(parts)

    def _intrinsic_stack_summary(self, rl_agent) -> str:
        """Banner block for the IntrinsicCriticStack reward terms + critic head.

        α / β axes are rendered by :meth:`_alpha_beta_summary`.
        """
        ic = getattr(rl_agent, "intrinsic_stack", None)
        if ic is None:
            alpha_dist = getattr(rl_agent, "alpha_dist", None)
            beta_dist = getattr(rl_agent, "beta_dist", None)
            return (
                f"NOT CONSTRUCTED (alpha_dist.max_value="
                f"{alpha_dist.max_value if alpha_dist else '?'}, "
                f"beta_dist.max_value="
                f"{beta_dist.max_value if beta_dist else '?'})"
            )
        parts = []
        terms = []
        if ic.e3b_coeff > 0:
            feat_str = ic.e3b_feature_key
            comp_dim = getattr(ic, "e3b_feature_dim", None)
            if comp_dim is not None:
                feat_str = f"{feat_str} (composite_dim={comp_dim})"
            terms.append(
                f"E3B(coeff={ic.e3b_coeff}, lambda={ic.e3b_lambda}, feature={feat_str})"
            )
        if getattr(ic, "ngu_coeff", 0.0) > 0:
            feat_str = ic.ngu_feature_key
            ngu_dim = getattr(ic, "ngu_feature_dim", None)
            if ngu_dim is not None:
                feat_str = f"{feat_str} (composite_dim={ngu_dim})"
            terms.append(
                f"NGU(coeff={ic.ngu_coeff}, k={ic.ngu_k}, feature={feat_str}, "
                f"c={ic.ngu_c:g}, eps={ic.ngu_kernel_eps:g}, "
                f"cluster={ic.ngu_cluster_distance:g}, "
                f"max_sim={ic.ngu_max_similarity:g})"
            )
        if ic.rnd_coeff_init > 0 or ic.rnd_coeff_final > 0:
            terms.append(
                f"RND(coeff {ic.rnd_coeff_init:g}->{ic.rnd_coeff_final:g} "
                f"over {ic.rnd_coeff_decay_steps} steps, obs={ic.rnd_obs_key})"
            )
        if ic.rldp_error_coeff > 0:
            terms.append(f"RLDP-error(coeff={ic.rldp_error_coeff})")
        parts.append(f"Reward terms: {' + '.join(terms) if terms else 'none'}")
        parts.append(
            f"  Encoders:   shared with actor (policy_ctx_dim={ic.policy_ctx_dim})"
        )
        critic_bits = [
            f"num_critics={ic.num_critics}",
            f"n_step={ic.n_step}",
            f"tau={ic.tau}",
            f"reward_mult={ic.reward_multiplier}",
            f"use_z_in_critic={ic.use_z_in_critic}",
        ]
        loss_w = getattr(rl_agent, "intrinsic_critic_loss_weight", None)
        if loss_w is not None:
            critic_bits.append(f"loss_weight={loss_w}")
        parts.append("  Critic:     " + ", ".join(critic_bits))
        ic_params = sum(p.numel() for p in ic.parameters())
        parts.append(f"  Parameters: {ic_params:,d}")
        return "\n    ".join(parts)

    def _policy_summary(self, rl_agent) -> str:
        parts = []
        # Phi / psi dims (TDJepa).
        phi_dim = getattr(rl_agent, "phi_dim", None)
        psi_dim = getattr(rl_agent, "psi_dim", None)
        if phi_dim is not None and psi_dim is not None:
            ortho = getattr(rl_agent, "fb_ortho_coeff", None)
            extras = [f"phi_dim={phi_dim}", f"psi_dim={psi_dim}"]
            if ortho is not None:
                extras.append(f"fb_ortho_coeff={ortho}")
            parts.append("SF heads: " + ", ".join(extras))
        # SF tstep / traj encoders.
        sf_tstep = getattr(rl_agent, "tstep_encoder", None)
        sf_traj = getattr(rl_agent, "traj_encoder", None)
        if sf_tstep is not None:
            parts.append(
                "  " + self._describe_tstep_encoder(sf_tstep, label="SF tstep (phi)")
            )
        if sf_traj is not None:
            parts.append(f"  SF traj (phi): {self._describe_traj_encoder(sf_traj)}")
        sf_tstep_b = getattr(rl_agent, "tstep_encoder_b", None)
        sf_traj_b = getattr(rl_agent, "traj_encoder_b", None)
        if sf_tstep_b is not None and sf_tstep_b is not sf_tstep:
            parts.append(
                "  " + self._describe_tstep_encoder(sf_tstep_b, label="SF tstep (psi)")
            )
        if sf_traj_b is not None and sf_traj_b is not sf_traj:
            parts.append(f"  SF traj (psi): {self._describe_traj_encoder(sf_traj_b)}")
        # Phi / psi MLP encoders (JEPA).
        phi_enc = getattr(rl_agent, "phi_encoder", None)
        if phi_enc is not None:
            parts.append(f"  Phi encoder MLP:  {self._describe_mlp(phi_enc)}")
        psi_enc = getattr(rl_agent, "psi_encoder", None)
        if psi_enc is not None:
            parts.append(f"  Psi encoder MLP:  {self._describe_mlp(psi_enc)}")
        # Phi / psi predictor MLPs (ensembled EinMix).
        phi_pred = getattr(rl_agent, "phi_predictor", None)
        if phi_pred is not None:
            parts.append(
                f"  Phi predictor:    {self._describe_jepa_predictor(phi_pred)}"
            )
        psi_pred = getattr(rl_agent, "psi_predictor", None)
        if psi_pred is not None:
            parts.append(
                f"  Psi predictor:    {self._describe_jepa_predictor(psi_pred)}"
            )
        # Policy-specific tstep / traj encoders.
        pi_tstep = getattr(rl_agent, "tstep_encoder_pi", None)
        if pi_tstep is not None:
            parts.append(
                "  " + self._describe_tstep_encoder(pi_tstep, label="Policy tstep")
            )
            pi_tstep_base = getattr(pi_tstep, "base", None)
            if pi_tstep_base is not None:
                parts.append(f"  Policy tstep MLP: {self._describe_mlp(pi_tstep_base)}")
        pi_traj = getattr(rl_agent, "traj_encoder_pi", None)
        if pi_traj is not None:
            parts.append(f"  Policy traj:   {self._describe_traj_encoder(pi_traj)}")
        # Extrinsic-reward branch: tstep + traj encoders, β-gated before
        # concat into the actor MLP. Only present when the β preset is not
        # ``beta/off`` (TDJepa.__init__ auto-skips construction when
        # beta_dist.max_value == 0).
        ext_tstep = getattr(rl_agent, "extrinsic_tstep_encoder", None)
        ext_traj = getattr(rl_agent, "extrinsic_traj_encoder", None)
        if ext_tstep is not None:
            parts.append(
                "  " + self._describe_tstep_encoder(ext_tstep, label="Ext tstep")
            )
            ext_tstep_base = getattr(ext_tstep, "base", None)
            if ext_tstep_base is not None:
                parts.append(
                    f"  Ext tstep MLP:    {self._describe_mlp(ext_tstep_base)}"
                )
        if ext_traj is not None:
            parts.append(f"  Ext traj:      {self._describe_traj_encoder(ext_traj)}")
        # Actor head MLP shape + post-concat input breakdown.
        actor = getattr(rl_agent, "actor", None)
        if actor is not None:
            actor_base = getattr(actor, "base", None)
            if actor_base is not None:
                parts.append(f"  Actor MLP:        {self._describe_mlp(actor_base)}")
            pi_traj_dim = getattr(
                getattr(rl_agent, "traj_encoder_pi", None), "emb_dim", None
            )
            psi_dim = getattr(rl_agent, "psi_dim", None)
            ab_enc = getattr(rl_agent, "alpha_beta_enc_dim", None)
            ext_ctx = getattr(rl_agent, "ext_ctx_dim", None)
            actor_state_dim = getattr(actor, "state_dim", None)
            if all(
                x is not None
                for x in (pi_traj_dim, psi_dim, ab_enc, ext_ctx, actor_state_dim)
            ):
                parts.append(
                    f"  Actor input:      policy_ctx({pi_traj_dim}) + z({psi_dim}) + "
                    f"αβ_enc({ab_enc}) + ext_ctx({ext_ctx}) = {actor_state_dim}"
                )
        # Q_I critic head MLP + post-concat input breakdown (mirrors actor;
        # critic appends z at the end instead of after policy_ctx).
        ic = getattr(rl_agent, "intrinsic_stack", None)
        if ic is not None:
            crit_state_dim = None
            try:
                crit_state_dim = ic.critics.state_dim
            except AttributeError:
                pass
            policy_ctx_dim = getattr(ic, "policy_ctx_dim", None)
            ab_enc = getattr(rl_agent, "alpha_beta_enc_dim", None)
            ext_ctx = getattr(rl_agent, "ext_ctx_dim", None)
            z_in_critic = getattr(ic, "use_z_in_critic", False)
            z_dim = getattr(ic, "z_dim", 0) if z_in_critic else 0
            if all(
                x is not None for x in (policy_ctx_dim, ab_enc, ext_ctx, crit_state_dim)
            ):
                parts.append(
                    f"  Q_I critic input: policy_ctx({policy_ctx_dim}) + αβ_enc({ab_enc}) + "
                    f"ext_ctx({ext_ctx}) + z({z_dim}) = {crit_state_dim} "
                    f"(NCritics x{ic.num_critics})"
                )
        if getattr(rl_agent, "zero_policy_ctx", False):
            parts.append(
                "  [DEBUG] zero_policy_ctx=True — policy_ctx AND z are "
                "zeroed at actor + Q_I critic inputs (TDJepa SF / BLR "
                "features hidden downstream; effectively trains AMAGO on "
                "ext_ctx + α,β fingerprint at α=β=1)."
            )
        # Loss / training knobs.
        loss_bits = []
        gammas = getattr(rl_agent, "gammas", None)
        if gammas is not None:
            try:
                gvals = [f"{float(g):.4g}" for g in gammas.flatten().tolist()]
                loss_bits.append(f"gammas={gvals[0] if len(gvals) == 1 else gvals}")
            except Exception:
                loss_bits.append(f"gammas={gammas}")
        for k in (
            "tau",
            "online_coeff",
            "offline_coeff",
            "sf_loss_weight",
            "reward_regression_coeff",
            "z_goal_ratio",
            "z_reward_ratio",
            "num_critics",
            "num_critics_td",
            "normalize_q",
            "detach_actor",
        ):
            v = getattr(rl_agent, k, None)
            if v is not None:
                loss_bits.append(f"{k}={v}")
        if loss_bits:
            parts.append("  Loss/train: " + ", ".join(loss_bits))
        # Actor parameter count.
        actor = getattr(rl_agent, "actor", None)
        if actor is not None:
            actor_params = sum(p.numel() for p in actor.parameters())
            parts.append(f"  Actor params: {actor_params:,d}")
        return "\n    ".join(parts)

    def _env_knobs_summary(self, rl_agent, dyn) -> str:
        """Surface the dimensions that env overlays are expected to tune.

        Dynamics output dims are the headline cross-env knobs:
          - rldp_dim          — DynamicsEncoder.dynamics_output_dim
          - idm_dim           — DynamicsEncoder.idm_dim
          - dyn_tstep_emb_dim — dynamics/FFTstepEncoder.d_output (== the
            ``dyn_tstep_emb`` obs key consumed by the policy side-channel
            and as one term in the E3B composite feature)
        TDJepa SF head dims (phi/psi) and trial windows (max_seq_len /
        ac_seq_len) are also tuned per env.
        """
        rldp_dim = getattr(dyn, "rldp_dim", None)
        idm_dim = getattr(dyn, "idm_dim", None)
        dyn_tstep_dim = getattr(dyn, "dyn_tstep_dim", None)
        phi_dim = getattr(rl_agent, "phi_dim", None)
        psi_dim = getattr(rl_agent, "psi_dim", None)
        ac_seq_len = rl_agent.ac_seq_len
        lines = []
        dyn_bits = []
        if rldp_dim is not None:
            dyn_bits.append(f"rldp_dim={rldp_dim}")
        if idm_dim is not None:
            dyn_bits.append(f"idm_dim={idm_dim}")
        if dyn_tstep_dim is not None:
            dyn_bits.append(f"dyn_tstep_emb_dim={dyn_tstep_dim}")
        if dyn_bits:
            lines.append("Dynamics outputs:    " + ", ".join(dyn_bits))
        sf_bits = []
        if phi_dim is not None:
            sf_bits.append(f"phi_dim={phi_dim}")
        if psi_dim is not None:
            sf_bits.append(f"psi_dim={psi_dim}")
        if sf_bits:
            lines.append("TDJepa SF heads:     " + ", ".join(sf_bits))
        win_bits = [
            f"max_seq_len={self.max_seq_len}",
            f"ac_seq_len={ac_seq_len}",
        ]
        lines.append("Trial windows:       " + ", ".join(win_bits))
        return "\n    ".join(lines)

    def _clip_groups_summary(self) -> str:
        """Render per-group clip type + threshold + param count.

        Mirrors the partition applied by ``train_step``: ``dynamics`` /
        ``amago`` / ``rnd`` are norm-clipped at ``self.clip_norm``;
        ``tdjepa_core`` is per-element value-clipped at ``self.grad_clip``.
        Param counts confirm the disjoint partition is non-empty for each
        active subsystem (e.g. dynamics shows 0 under ``dynamics/noop``;
        rnd shows 0 when intrinsic_stack is disabled).
        """
        groups = self.policy.clip_param_groups()
        lines = []
        norm_thr = self.clip_norm if self.clip_norm and self.clip_norm > 0 else None
        val_thr = self.grad_clip if self.grad_clip and self.grad_clip > 0 else None
        for name, params in (
            ("dynamics", groups.get("dynamics", [])),
            ("amago", groups.get("amago", [])),
            ("rnd", groups.get("rnd", [])),
        ):
            n_params = sum(p.numel() for p in params if p.requires_grad)
            if norm_thr is not None:
                clip_desc = f"clip_grad_norm_({norm_thr})"
            else:
                clip_desc = "clip_grad_norm_ disabled (clip_norm <= 0)"
            lines.append(f"{name:<12} {clip_desc} on {n_params:,d} params")
        core_params = groups.get("tdjepa_core", [])
        n_core = sum(p.numel() for p in core_params if p.requires_grad)
        if val_thr is not None:
            core_desc = f"clip_grad_value_({val_thr})"
        else:
            core_desc = "clip_grad_value_ disabled (grad_clip <= 0)"
        lines.append(f"{'tdjepa_core':<12} {core_desc} on {n_core:,d} params")
        return "\n    ".join(lines)

    def _baseline_preset_header(self) -> str:
        """Render the baseline-preset header, including the resolved gin ladder.

        Both ``self._baseline_preset`` and ``self._gin_ladder`` are launcher-
        supplied; if absent (notebook / ad-hoc caller) we render a clean
        ``<not provided>`` line so the absence is visible rather than silent.
        """
        if self._baseline_preset is None and not self._gin_ladder:
            return "Baseline preset: <not provided by launcher>"
        lines = [f"Baseline preset: {self._baseline_preset or '<unset>'}"]
        if self._gin_ladder:
            lines.append("Gin ladder:")
            for c in self._gin_ladder:
                lines.append(f"  {c}")
        return "\n    ".join(lines)

    def summary(self, env_summary: str) -> None:
        total_params = utils.count_params(self.policy)
        rl_agent = self.policy.rl_agent
        rl_params = sum(p.numel() for p in rl_agent.trainable_params)
        dyn = self.policy.dynamics_learner

        baseline_header = self._baseline_preset_header()
        env_knobs = self._env_knobs_summary(rl_agent, dyn)
        seq_summary = self._seq_lengths_summary(rl_agent, dyn)
        dyn_summary = self._dynamics_summary(dyn)
        alpha_beta_summary = self._alpha_beta_summary(rl_agent)
        intrinsic_stack_summary = self._intrinsic_stack_summary(rl_agent)
        policy_summary = self._policy_summary(rl_agent)
        clip_groups_summary = self._clip_groups_summary()

        bs = getattr(self, "batch_size", "?")
        lr = getattr(self, "learning_rate", "?")

        self.accelerator.print(
            f"""
  ══════════════════════════════════════════════════════════
  BilevelExperiment Summary
  ══════════════════════════════════════════════════════════

  {baseline_header}

  Env-tuned dimensions (configs/env/<env>.gin):
    {env_knobs}

  Sequence Lengths:
    {seq_summary}

  Dynamics Learner:
    {dyn_summary}

  α-β configuration:
    {alpha_beta_summary}

  Intrinsic Critic Stack:
    {intrinsic_stack_summary}

  Policy (TDJepa):
    {policy_summary}

  Parameters:
    RL Agent: {rl_params:,d} (trainable)
    Total:    {total_params:,d}

  Grad clipping (per disjoint subgraph):
    {clip_groups_summary}

  Training:
    batch_size:     {bs}
    learning_rate:  {lr}
    checkpoint_dir: {self.ckpt_dir}

  Environment:
    {env_summary}

  ══════════════════════════════════════════════════════════
"""
        )

    # ── Per-episode alpha resampling hook (collection-time only) ──────────

    def on_interact_step(self, obs) -> None:
        """Resample ``agent._inference_alpha`` (and ``_inference_beta``) for
        actors whose ``episode_id`` just changed, drawing fresh values from
        ``agent.alpha_dist`` / ``agent.beta_dist``.

        Gated by ``self._resample_alpha_per_episode_active`` so only the
        ``self.interact(...)`` call inside ``collect_new_training_data`` is
        affected; evaluation (``_interact_with_inference``) is untouched.
        """
        if not self._resample_alpha_per_episode_active:
            return
        agent = self.policy.rl_agent
        ep_id = obs["episode_id"].squeeze(1)
        if self._prev_ep_id is None:
            self._prev_ep_id = ep_id.clone()
            return
        changed = ep_id != self._prev_ep_id
        if changed.any():
            # Route through propose_*_for_collection rather than sampling the
            # dists directly so ``pin_pure_extrinsic_fraction`` actors stay
            # pinned at 1.0 across episode boundaries within a meta-trial.
            fresh = agent.propose_alpha_for_collection(
                self.parallel_actors, self.DEVICE
            )
            agent._inference_alpha = torch.where(
                changed.unsqueeze(-1),
                fresh,
                agent._inference_alpha,
            )
            if agent._inference_beta is not None:
                fresh_b = agent.propose_beta_for_collection(
                    self.parallel_actors, self.DEVICE
                )
                agent._inference_beta = torch.where(
                    changed.unsqueeze(-1),
                    fresh_b,
                    agent._inference_beta,
                )
        self._prev_ep_id = ep_id.clone()

    # ── Online collection ─────────────────────────────────────────────────

    def collect_new_training_data(self):
        if self.num_collection_batches is None:
            super().collect_new_training_data()
            return

        from metabfm.rl.zero_shot_rl import pairwise_cosine

        agent = self.policy.rl_agent
        self.policy.eval()

        rounds = range(self.num_collection_batches)
        if self.verbose:
            rounds = tqdm(
                rounds,
                desc="Collection Rounds",
                total=self.num_collection_batches,
                leave=False,
                colour="cyan",
            )

        all_returns, all_specials = [], []
        for r in rounds:
            z = agent.propose_z_for_collection(self.parallel_actors, self.DEVICE)
            agent._inference_z = z
            agent._inference_alpha = agent.propose_alpha_for_collection(
                self.parallel_actors, self.DEVICE
            )
            agent._inference_beta = agent.propose_beta_for_collection(
                self.parallel_actors, self.DEVICE
            )

            self.train_envs.reset()
            self.hidden_state = None

            if self.verbose:
                cos_sim = pairwise_cosine(z)
                alpha = agent._inference_alpha
                alpha_msg = ""
                if alpha is not None:
                    frac_nonzero = (alpha > 0).float().mean().item()
                    alpha_msg = f", alpha mean={alpha.mean().item():.3f} nonzero={frac_nonzero:.0%}"
                self.accelerator.print(
                    f"  Round {r}: reset envs, sampled {self.parallel_actors} z vectors "
                    f"(pairwise cosine {cos_sim:.3f}{alpha_msg})"
                )

            self._prev_ep_id = None
            self._resample_alpha_per_episode_active = self.resample_alpha_per_episode
            try:
                self.hidden_state, (returns, specials) = self.interact(
                    self.train_envs,
                    self.steps_per_round,
                    hidden_state=self.hidden_state,
                    sample=self.sample_actions_train,
                )
            finally:
                self._resample_alpha_per_episode_active = False
            all_returns.extend(returns)
            all_specials.extend(specials)
            utils.call_async_env(self.train_envs, "save_finished_trajs")

        logs = self.policy_metrics(all_returns, all_specials)
        logs.update(self._dispersion_stats(all_returns, all_specials))
        logs["z_pairwise_cosine"] = pairwise_cosine(z)
        logs["num_rounds"] = self.num_collection_batches
        alpha = agent._inference_alpha
        if alpha is not None:
            logs["alpha_mean"] = alpha.mean().item()
            logs["alpha_max"] = alpha.max().item()
            logs["alpha_frac_nonzero"] = (alpha > 0).float().mean().item()
        beta = agent._inference_beta
        if beta is not None:
            logs["beta_mean"] = beta.mean().item()
            logs["beta_max"] = beta.max().item()
            logs["beta_frac_nonzero"] = (beta > 0).float().mean().item()
        self.log(logs, key="exploration")

    @staticmethod
    def _dispersion_stats(all_returns, all_specials) -> dict:
        """Min/Max/P25/P75 of per-trial Total Return and Total Successes across actors.

        Computed over the flat list of per-trial scalars contributed by all
        parallel actors across all collection rounds, to expose whether
        different actors are producing different returns/success under their
        own (z, alpha) draws.
        """
        returns_by_env: dict[str, list[float]] = {}
        for ret in all_returns:
            for env_name, scores in ret.data.items():
                returns_by_env.setdefault(env_name, []).extend(scores)

        successes_by_env: dict[str, list[float]] = {}
        for spe in all_specials:
            for env_name, specials_dict in spe.data.items():
                if "Total Successes" in specials_dict:
                    successes_by_env.setdefault(env_name, []).extend(
                        specials_dict["Total Successes"]
                    )

        out: dict = {}

        def _emit(label: str, name: str, vals: list[float]) -> None:
            if not vals:
                return
            arr = np.asarray(vals, dtype=np.float64)
            out[f"Min {label} in {name}"] = float(arr.min())
            out[f"Max {label} in {name}"] = float(arr.max())
            out[f"P25 {label} in {name}"] = float(np.percentile(arr, 25))
            out[f"P75 {label} in {name}"] = float(np.percentile(arr, 75))

        for name, vals in returns_by_env.items():
            _emit("Total Return", name, vals)
        for name, vals in successes_by_env.items():
            _emit("Total Successes", name, vals)
        return out

    def _run_eval_phase(
        self,
        envs,
        timesteps: int,
        alpha_schedule,
        log_key: str,
        sample: bool,
        beta_schedule=None,
    ) -> dict:
        """Run one evaluation *phase* (one (α, β) regime) against ``envs``.

        * **trial** — one task instance spanning exactly ``k`` episodes; the
          env auto-resets to a new ``task_id`` at the end of each trial.
        * **round** — one ``(reset inference z, run one trial, record
          returns)`` per actor.  ``on_session_end`` wipes BLR state on each
          ``done`` so posteriors do not leak across trials.

        Runs ``timesteps // steps_per_trial`` rounds per actor via
        ``_interact_with_inference``.  ``beta_schedule`` defaults to
        constant β=0.
        """
        from metabfm.rl.zero_shot_rl import ConstantAlpha, ConstantSchedule

        if beta_schedule is None:
            beta_schedule = ConstantSchedule(ConstantAlpha(0.0))

        policy = self.accelerator.unwrap_model(self.policy_aclr)
        agent = policy.rl_agent
        inference_mode = agent.inference_mode
        n_actors = self.parallel_actors

        inference_mode.init_z(agent, n_actors=n_actors, device=self.DEVICE)

        envs.reset()

        _, (returns, specials), ir_tensor = self._interact_with_inference(
            envs,
            timesteps,
            inference_mode,
            alpha_schedule=alpha_schedule,
            beta_schedule=beta_schedule,
            sample=sample,
            log_key=log_key,
        )

        logs_per_process = self.policy_metrics(returns, specials=specials)
        logs_global = utils.avg_over_accelerate(logs_per_process)
        # Per-episode-in-trial breakdown from keys like
        # "Average Episode <N> <metric> in <env_name>", where N indexes the
        # episode within a k-episode meta-trial.
        by_ep, by_ep_per_task = self._per_episode_summary(logs_global)
        if self.verbose:
            ret_key = "Average Total Return (Across All Env Names)"
            if ret_key in logs_global:
                self.accelerator.print(
                    f"{log_key} Average Return: {logs_global[ret_key]:.3f}"
                )
            for ep_idx in sorted(by_ep):
                parts = ", ".join(
                    f"{name}={val:.3f}" for name, val in sorted(by_ep[ep_idx].items())
                )
                self.accelerator.print(f"  {log_key} Ep{ep_idx}: {parts}")
        self._log_eval_phase(logs_global, log_key, by_ep, by_ep_per_task, ir_tensor)
        return logs_global

    @staticmethod
    def _per_episode_summary(
        logs_global: dict,
    ) -> tuple[dict[int, dict[str, float]], dict[str, dict[int, dict[str, float]]]]:
        """Extract per-episode-in-trial metrics from the policy-metrics dict.

        Source keys have the form
        ``"Average <Trial|Episode> <N> <metric> (Across All Env Names)"`` or
        ``"... in <env_name>"``; ``<metric>`` is any non-empty string (e.g.
        ``"Return"``, ``"Oracle Normalized Return"``, ``"DMC Score (run)"``).

        Returns two views:
          * ``aggregated``: ``{ep_idx: {metric: value}}`` — the
            "(Across All Env Names)" variant if present, else the mean of
            the per-env_name entries.
          * ``per_task``: ``{env_name: {ep_idx: {metric: value}}}``.
        """
        import re

        across_pat = re.compile(
            r"^Average (?:Trial|Episode) (\d+) (.+?) \(Across All Env Names\)$"
        )
        per_env_pat = re.compile(r"^Average (?:Trial|Episode) (\d+) (.+?) in (.+)$")
        aggregated: dict[int, dict[str, float]] = {}
        per_task: dict[str, dict[int, dict[str, float]]] = {}
        per_env_for_agg: dict[tuple[int, str], list[float]] = {}

        for k, v in logs_global.items():
            m = across_pat.match(k)
            if m is not None:
                ep_idx = int(m.group(1))
                metric = m.group(2)
                aggregated.setdefault(ep_idx, {})[metric] = float(v)
                continue
            m = per_env_pat.match(k)
            if m is not None:
                ep_idx = int(m.group(1))
                metric = m.group(2)
                env_name = m.group(3)
                per_task.setdefault(env_name, {}).setdefault(ep_idx, {})[metric] = (
                    float(v)
                )
                per_env_for_agg.setdefault((ep_idx, metric), []).append(float(v))

        for (ep_idx, metric), vals in per_env_for_agg.items():
            if metric in aggregated.get(ep_idx, {}):
                continue  # across-all variant already captured
            aggregated.setdefault(ep_idx, {})[metric] = float(np.mean(vals))
        return aggregated, per_task

    def _log_eval_phase(
        self,
        logs_global: dict,
        log_key: str,
        by_ep: dict[int, dict[str, float]],
        by_ep_per_task: dict[str, dict[int, dict[str, float]]],
        ir_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        """Log eval-phase scalars + per-episode-in-trial line charts to wandb.

        Charts emitted per phase (wandb only):
          * ``{log_key}/per_episode/<metric>_curve`` — averaged across tasks.
          * ``{log_key}/per_episode/<metric>_by_task`` — one line per
            env_name; skipped when there's only one task.
          * ``{log_key}/intrinsic_reward_curve`` — mean intrinsic reward vs
            timestep, if ``ir_tensor`` is provided.
        """
        # self.log writes to both history.jsonl and wandb scalars.
        self.log(logs_global, key=log_key)

        if not self.log_to_wandb:
            return

        # Build wandb-only chart objects (Tables / line_series) that can't go
        # through self.log, then push them alongside x-axis metrics.
        wandb_dict: dict = {}

        if self.accelerator.is_main_process:
            import wandb

            # Discover the set of metrics actually present (env-dependent).
            agg_metrics: set[str] = set()
            for ep_data in by_ep.values():
                agg_metrics.update(ep_data.keys())
            per_task_metrics: set[str] = set()
            for ep_map in by_ep_per_task.values():
                for ep_data in ep_map.values():
                    per_task_metrics.update(ep_data.keys())

            for metric in sorted(agg_metrics):
                rows = [
                    [ep, by_ep[ep][metric]]
                    for ep in sorted(by_ep)
                    if metric in by_ep[ep]
                ]
                if len(rows) < 2:
                    continue
                table = wandb.Table(data=rows, columns=["episode", metric])
                wandb_dict[f"{log_key}/per_episode/{metric}_curve"] = wandb.plot.line(
                    table,
                    "episode",
                    metric,
                    title=f"{log_key} — {metric} vs Episode (in trial)",
                )

            # Per-task multi-line charts. Skip if there's only one task (the
            # aggregated chart already shows it).
            if len(by_ep_per_task) > 1:
                for metric in sorted(per_task_metrics):
                    task_names = sorted(by_ep_per_task.keys())
                    xs, ys, keys = [], [], []
                    for env_name in task_names:
                        ep_map = by_ep_per_task[env_name]
                        sorted_eps = sorted(ep for ep in ep_map if metric in ep_map[ep])
                        if len(sorted_eps) < 2:
                            continue
                        xs.append(sorted_eps)
                        ys.append([ep_map[ep][metric] for ep in sorted_eps])
                        keys.append(env_name)
                    if len(keys) < 1:
                        continue
                    wandb_dict[f"{log_key}/per_episode/{metric}_by_task"] = (
                        wandb.plot.line_series(
                            xs=xs,
                            ys=ys,
                            keys=keys,
                            title=(f"{log_key} — {metric} vs Episode by task"),
                            xname="episode",
                        )
                    )

            if ir_tensor is not None and ir_tensor.shape[0] > 0:
                ir_np = ir_tensor.numpy()  # (T, n_actors)
                ir_mean = ir_np.mean(axis=1)
                T = ir_mean.shape[0]
                step_stride = max(1, T // 500)
                steps = list(range(0, T, step_stride))
                wandb_dict[f"{log_key}/intrinsic_reward_curve"] = (
                    wandb.plot.line_series(
                        xs=[steps],
                        ys=[ir_mean[::step_stride].tolist()],
                        keys=["mean"],
                        title=f"{log_key} — Intrinsic Reward vs Timestep",
                        xname="timestep",
                    )
                )

        if wandb_dict:
            wandb_dict.update(self.x_axis_metrics())
            self.accelerator.log(wandb_dict)

    def _interact_with_inference(
        self,
        envs,
        timesteps,
        inference_mode,
        alpha_schedule,
        sample=True,
        log_key: str = "eval",
        beta_schedule=None,
    ):
        """Custom rollout loop that calls InferenceMode on_step / on_session_end hooks.

        Mirrors ``Experiment.interact`` but inserts per-step BLR updates and
        z-resampling between the action computation and the env step.
        ``beta_schedule`` defaults to constant β=0.

        Returns ``(hidden_state, (return_history, special_history),
        ir_tensor)`` where ``ir_tensor`` is the ``(T, n_actors)`` intrinsic
        reward trace (or None).
        """
        from metabfm.rl.zero_shot_rl import ConstantAlpha, ConstantSchedule

        if beta_schedule is None:
            beta_schedule = ConstantSchedule(ConstantAlpha(0.0))

        # As in AMAGO's ``Experiment.interact``, the rollout primitive owns
        # the eval-mode guard (the transformer asserts ``not self.training``,
        # e.g. when ``evaluate_test`` runs after ``learn()``).  ``learn()``
        # switches back to ``train()`` before the next gradient step.
        self.policy.eval()
        policy = self.accelerator.unwrap_model(self.policy_aclr)
        agent = policy.rl_agent
        n_actors = self.parallel_actors

        hidden_state = policy.traj_encoder.init_hidden_state(n_actors, self.DEVICE)

        def get_t():
            par = utils.call_async_env(envs, "current_timestep")
            _obs, _rl2s, _time = [], [], []
            for _o, _r, _t in par:
                _obs.append(_o)
                _rl2s.append(_r)
                _time.append(_t)
            _obs = utils.stack_list_array_dicts(_obs, axis=0, cat=True)
            _rl2s = np.concatenate(_rl2s, axis=0)
            _time = np.concatenate(_time, axis=0)
            _obs = {
                k: torch.from_numpy(v).to(self.DEVICE).unsqueeze(1)
                for k, v in _obs.items()
            }
            _rl2s = torch.from_numpy(_rl2s).to(self.DEVICE).unsqueeze(1)
            _time = torch.from_numpy(_time).to(self.DEVICE).unsqueeze(1)
            return _obs, _rl2s, _time

        obs, rl2s, time_idxs = get_t()
        cached_phi = None
        prev_reward = None

        utils.call_async_env(envs, "reset_stats")

        if self.verbose:
            iter_ = tqdm(
                range(timesteps),
                desc=f"Eval Phase [{log_key}]",
                total=timesteps,
                leave=False,
                colour="yellow",
            )
        else:
            iter_ = range(timesteps)

        ir_trace: list[torch.Tensor] = []

        for _t in iter_:
            ep_id = obs["episode_id"].squeeze(1)
            agent._inference_alpha = alpha_schedule(ep_id, n_actors, self.DEVICE)
            agent._inference_beta = beta_schedule(ep_id, n_actors, self.DEVICE)

            with torch.no_grad():
                actions, hidden_state = policy.get_actions(
                    obs=obs,
                    rl2s=rl2s,
                    time_idxs=time_idxs,
                    hidden_state=hidden_state,
                    sample=sample,
                )

            if "intrinsic_reward" in obs:
                ir_trace.append(obs["intrinsic_reward"][:, 0, 0].cpu())

            cached_phi = agent.extract_phi_for_blr(actions=actions)

            # Convention B: pair the freshly-extracted ψ(s_t) with the previous
            # transition's reward r_{t-1} (= the reward received entering s_t),
            # which matches the SF Bellman target <ψ(s_{t+1}), z> ≈ r_t.  Action
            # a_t was already selected with the prior posterior; the new BLR
            # update applies starting at iter t+1 (one-step staleness).
            if prev_reward is not None:
                r_tensor = torch.from_numpy(prev_reward).float().to(self.DEVICE)
                inference_mode.on_step(agent, cached_phi, r_tensor, obs)

            actions_np = actions.squeeze(1).cpu().numpy()
            _, prev_reward, terminated, truncated, _ = envs.step(actions_np)
            done = terminated | truncated
            if done.ndim == 2:
                done = done.squeeze(1)
            if prev_reward.ndim == 2:
                prev_reward = prev_reward.squeeze(1)

            obs, rl2s, time_idxs = get_t()
            hidden_state = policy.traj_encoder.reset_hidden_state(hidden_state, done)

            if done.any():
                done_t = torch.from_numpy(done).to(self.DEVICE)
                inference_mode.on_session_end(agent, done_t)

        ir_tensor = torch.stack(ir_trace, dim=0) if ir_trace else None  # (T, n_actors)

        return_history = utils.call_async_env(envs, "return_history")
        special_history = utils.call_async_env(envs, "special_history")
        return hidden_state, (return_history, special_history), ir_tensor

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate_val(self) -> None:
        """Up-to-three-phase ZSRL evaluation on ``self.val_envs``.

        * ``"val"``: α=0, β=0 — the zero-shot rollout all baselines log.
        * ``"val-alpha"``: ``val_alpha_schedule`` at β=0; skipped when
          ``val_alpha_schedule.max_value <= 0`` (e.g. ``dyncs_zsrl``,
          ``pure_extrinsic``).
        * ``"val-beta"``: α=1 with ``val_beta_schedule``; skipped when
          ``val_beta_schedule.max_value <= 0``.

        Each phase resets envs and re-initializes the task latent ``z``
        (see :meth:`_run_eval_phase`).
        """
        from metabfm.rl.zero_shot_rl import ConstantAlpha, ConstantSchedule

        policy = self.accelerator.unwrap_model(self.policy_aclr)
        agent = policy.rl_agent

        zero_alpha = ConstantSchedule(ConstantAlpha(0.0))
        zero_beta = ConstantSchedule(ConstantAlpha(0.0))
        if getattr(self, "allow_val", True):
            self._run_eval_phase(
                self.val_envs,
                self.val_timesteps_per_epoch,
                alpha_schedule=zero_alpha,
                log_key="val",
                sample=self.sample_actions_val,
            )
        if agent.val_alpha_schedule.max_value > 0 and getattr(
            self, "allow_val_alpha", True
        ):
            self._run_eval_phase(
                self.val_envs,
                self.val_timesteps_per_epoch,
                alpha_schedule=agent.val_alpha_schedule,
                beta_schedule=zero_beta,
                log_key="val-alpha",
                sample=self.sample_actions_val,
            )
        if agent.val_beta_schedule.max_value > 0 and getattr(
            self, "allow_val_beta", True
        ):
            one_alpha = ConstantSchedule(ConstantAlpha(1.0))
            self._run_eval_phase(
                self.val_envs,
                self.val_timesteps_per_epoch,
                alpha_schedule=one_alpha,
                beta_schedule=agent.val_beta_schedule,
                log_key="val-beta",
                sample=self.sample_actions_val,
            )

    def evaluate_test(
        self,
        make_test_env,
        timesteps: int,
        render: bool = False,
        save_trajs_to: Optional[str] = None,
        episodes: Optional[int] = None,
        log_tag: Optional[str] = None,
        phase_prefix: str = "test",
    ) -> dict:
        """Up-to-three-phase ZSRL evaluation on a user-supplied test env callable.

        Builds the env the same way as AMAGO's ``Experiment.evaluate_test``
        (``async`` / ``sync`` / ``already_vectorized``), then runs the same
        phases as :meth:`evaluate_val`, logged under ``"{phase_prefix}"`` /
        ``"{phase_prefix}-alpha"`` / ``"{phase_prefix}-beta"``.

        ``log_tag``: optional suffix appended to every wandb log key as
        ``"{phase}/{log_tag}"`` (e.g. ``test/train_split/run``); without it,
        multiple ``evaluate_test`` calls in one epoch overwrite each other in
        wandb.  The returned dict keys omit the tag.

        ``phase_prefix``: ``"test"`` by default; pass ``"val"`` for
        multi-setup validation sweeps.

        ``render``, ``save_trajs_to`` and ``episodes`` are unused and kept
        for signature compatibility with the base class.

        Returns ``{short_phase_key -> logs_global}`` for every phase that ran.
        """
        from metabfm.rl.zero_shot_rl import ConstantAlpha, ConstantSchedule
        from amago.envs.env_utils import AlreadyVectorizedEnv

        del render, save_trajs_to, episodes

        policy = self.accelerator.unwrap_model(self.policy_aclr)
        agent = policy.rl_agent

        def wrap(m):
            return lambda: SequenceWrapper(m(), save_trajs_to=None, save_every=None)

        def _keys(suffix: str = "") -> tuple[str, str]:
            """Return (short_dict_key, wandb_log_key) for a phase suffix."""
            short = f"{phase_prefix}{suffix}"
            wandb_key = f"{short}/{log_tag}" if log_tag else short
            return short, wandb_key

        par_kwargs: dict = {}
        if self.env_mode == "already_vectorized":
            Par = AlreadyVectorizedEnv
            env_list = [wrap(make_test_env)]
        else:
            if isinstance(make_test_env, Iterable):
                assert len(make_test_env) == self.parallel_actors
                env_list = [wrap(f) for f in make_test_env]
            else:
                env_list = [wrap(make_test_env) for _ in range(self.parallel_actors)]
            if self.env_mode == "async":
                Par = gym.vector.AsyncVectorEnv
                par_kwargs = dict(context=self.async_env_mp_context)
            else:
                Par = DummyAsyncVectorEnv
        test_envs = Par(env_list, **par_kwargs)

        zero_alpha = ConstantSchedule(ConstantAlpha(0.0))
        zero_beta = ConstantSchedule(ConstantAlpha(0.0))
        # Collect every phase's logs so callers (e.g. ``run_checkpoint_eval``)
        # can persist results without scraping wandb.
        all_logs: dict[str, dict] = {}
        if getattr(self, "allow_val", True):
            short, wandb_key = _keys("")
            all_logs[short] = self._run_eval_phase(
                test_envs,
                timesteps,
                alpha_schedule=zero_alpha,
                log_key=wandb_key,
                sample=self.sample_actions_val,
            )
        if agent.val_alpha_schedule.max_value > 0 and getattr(
            self, "allow_val_alpha", True
        ):
            short, wandb_key = _keys("-alpha")
            all_logs[short] = self._run_eval_phase(
                test_envs,
                timesteps,
                alpha_schedule=agent.val_alpha_schedule,
                beta_schedule=zero_beta,
                log_key=wandb_key,
                sample=self.sample_actions_val,
            )
        if agent.val_beta_schedule.max_value > 0 and getattr(
            self, "allow_val_beta", True
        ):
            one_alpha = ConstantSchedule(ConstantAlpha(1.0))
            short, wandb_key = _keys("-beta")
            all_logs[short] = self._run_eval_phase(
                test_envs,
                timesteps,
                alpha_schedule=one_alpha,
                beta_schedule=agent.val_beta_schedule,
                log_key=wandb_key,
                sample=self.sample_actions_val,
            )
        test_envs.close()
        return all_logs
