"""State-based DMC Remastered walker with randomized embodiment, wrapped as a
k-shot meta-RL trial.

``dmc_remastered``'s dynamics-generalization walker resamples torso/foot
geometry, knee/ankle joint ranges, motor damping/armature, and ground friction
from a ``dynamics_seed``. Observations are the raw 24-D proprioceptive state
(no pixels), so the agent must infer its body from interaction.

The physics is identical across reward functions (the registered task builders
differ only in ``move_speed``), and every reward is a pure function of
``physics``, so one rollout scores all modes and a dynamics seed is the same
body in every mode. ``reset(soft=True)`` restarts the episode without
recompiling the model, keeping the body fixed across the trial.

A trial is ``k`` inner episodes of ``max_episode_steps`` agent steps, ending
with ``terminated = truncated = True``: ``max_episode_steps * k + 1`` timesteps
including the initial reset observation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence

import gymnasium as gym
import numpy as np
from dm_control.utils import rewards as dmc_rewards

from amago.envs import AMAGO_ENV_LOG_PREFIX

from dmc_remastered import DYNAMICS_ENVS
from dmc_remastered.walker import (
    _CONTROL_TIMESTEP,
    _RUN_SPEED,
    _STAND_HEIGHT,
    _WALK_SPEED,
)
from dmc_remastered.wrapper import DMC_Remastered_Env, _flatten_obs


#: Crouch target band for the torso frame's world z. Full extension is 1.30 m for
#: *every* dynamics seed: ``body_shape`` resizes only the torso geom and the feet,
#: while the leg chain (thigh attach at -0.3, thigh 0.7, leg 0.5) is fixed in the
#: XML, so an absolute height threshold needs no per-body normalization. A squat at
#: knee -40 to -80 degrees reaches 0.48-0.71 m with the torso still vertical, so
#: this band is comfortably inside the reachable set.
#:
#: The band is deliberately two-sided. A one-sided "below 0.9" bound would be
#: maximized by lying flat at h ~ 0.05, making the task solvable by falling over.
_CROUCH_BOUNDS = (0.6, 0.9)
_CROUCH_MARGIN = 0.2

#: Target backflip rate, rad/s (~1.6 turns/s). Walk-trained policies exceed this
#: on ~3% of steps, so the target sits in the tail of the visited distribution
#: rather than outside it. Positive pitches the torso top backward.
_BACKFLIP_RATE = 10.0


@dataclass(frozen=True)
class RewardSpec:
    """A walker reward: a torso-height term optionally gated by a speed term.

    Args:
        move_speed: Signed speed gate. ``0`` leaves the height term ungated; the
            sign picks the rewarded travel direction, so a negative value is the
            same reward with the velocity axis flipped.
        move_margin: Width of the speed falloff. ``None`` uses the native
            ``|move_speed| / 2``, which is flat for every velocity on the wrong
            side of zero and so gives a reversed target no gradient where the
            policy operates. ``|move_speed|`` makes the term monotone across
            ``[-speed, +speed]``.
        height_bounds: Rewarded band for ``physics.torso_height()``. One-sided by
            default, matching the native standing term.
        height_margin: Width of the tolerance falloff outside ``height_bounds``.
        spin_rate: Signed target pitch rate in rad/s. When set, the reward is the
            rotation term alone: a flip is not upright, so blending in posture
            would pay the agent to abort the rotation halfway. Positive spins
            the top of the torso backward; see :data:`_BACKFLIP_RATE`.
        move_ramp: ``(v_zero, v_full, rest)``: kinked velocity ramp times the
            height-only standing gate (no uprightness term, no 1/6 floor), in
            [0, 1] and monotone from ``v_zero`` (0) through rest (``rest``) to
            ``v_full`` (1). Height-only gating matters because the uprightness
            blend pays a mid-step backward lean less than standing still. The
            rest plateau matters because test-time ``z`` is fit by regressing
            observed reward on successor features: a reward that is ~0
            everywhere the policy currently goes leaves nothing to fit, and
            inference falls back to forward walking (``rest = 0`` empirically
            increased forward velocity; ``rest = 0.5`` reversed it).
        leap: ``(h_lo, h_hi, vz_scale)``: posture-free launch reward
            ``clip((h - h_lo)/(h_hi - h_lo), 0, 1)**2 * clip(vz / vz_scale, 0, 1)``
            where ``vz`` is the root vertical velocity. In [0, 1] and exactly
            zero for every static pose and descending state, so only pushing
            the body upward pays. Pure height-based jump rewards fail because
            high-torso states are rare (~0.1% of visited steps) and inference
            lands on walking; upward launches have far more support, giving BLR
            per-step contrast.
    """

    move_speed: float = 0.0
    move_margin: Optional[float] = None
    height_bounds: tuple = (_STAND_HEIGHT, float("inf"))
    height_margin: float = _STAND_HEIGHT / 2
    spin_rate: Optional[float] = None
    move_ramp: Optional[tuple] = None
    leap: Optional[tuple] = None


#: ``walk_backward`` ramp, as ``(v_zero, v_full, rest)`` -- see
#: :attr:`RewardSpec.move_ramp`. 0 at +1 m/s, 0.5 at rest, 1 at -1 m/s, all
#: times the height gate. Judge it by the per-episode ``Mean Vel`` /
#: ``Frac Vel Backward`` diagnostics, not reward score: the score alone cannot
#: separate standing from backward travel.
_BACKWARD_RAMP: tuple = (float(_WALK_SPEED), -float(_WALK_SPEED), 0.5)

#: Reward functions scored on every rollout (see :func:`walker_rewards`). Each
#: lands in [0, 1] per control step (``backflip`` in [-1, 1]) so modes stay
#: comparable on the 0-1000 DMC scale.
WALKER_REWARD_MODES: Dict[str, RewardSpec] = {
    "stand": RewardSpec(),
    "walk": RewardSpec(move_speed=float(_WALK_SPEED)),
    "run": RewardSpec(move_speed=float(_RUN_SPEED)),
    "walk_backward": RewardSpec(move_ramp=_BACKWARD_RAMP),
    "crouch": RewardSpec(
        height_bounds=_CROUCH_BOUNDS, height_margin=_CROUCH_MARGIN
    ),
    "crouch_walk": RewardSpec(
        move_speed=float(_WALK_SPEED),
        height_bounds=_CROUCH_BOUNDS,
        height_margin=_CROUCH_MARGIN,
    ),
    "backflip": RewardSpec(spin_rate=_BACKFLIP_RATE),
    # Pays only while the body is both elevated and moving upward (see
    # RewardSpec.leap), so the integral is maximized by repeated launches.
    "jump": RewardSpec(leap=(0.9, 1.4, 2.0)),
}

#: Modes that reproduce a native ``PlanarWalker`` reward exactly. These drive
#: in-training validation, so adding a mode cannot silently change it.
WALKER_CORE_REWARD_MODES: List[str] = ["stand", "walk", "run"]

#: Randomization axes. ``DMCR_VARY``'s visual keys are omitted because
#: observations are state-based; they would only change the render.
DYNAMICS_VARY: List[str] = ["body_shape", "motors", "friction"]

#: Per-control-step reward is in [0, 1] in every DMC suite task, and the
#: canonical DMC episode is 1000 control steps, so scores are reported out of
#: 1000 regardless of our frame_skip / horizon choices.
DMC_SCORE_SCALE = 1000.0


def walker_rewards(physics) -> Dict[str, float]:
    """Score every mode in :data:`WALKER_REWARD_MODES` from one physics state.

    Reproduces :meth:`dmc_remastered.walker.PlanarWalker.get_reward` exactly for
    :data:`WALKER_CORE_REWARD_MODES`. The other modes:

    * ``walk_backward`` -- height-gated velocity ramp (:data:`_BACKWARD_RAMP`),
      in [0, 1]: 0 walking forward, 0.5 at rest, 1 backward at 1 m/s.
    * ``crouch`` / ``crouch_walk`` -- two-sided :data:`_CROUCH_BOUNDS` height
      band; the uprightness term keeps a crouch a squat rather than a faceplant.
    * ``backflip`` -- clipped backward pitch rate in [-1, 1], no posture term
      (foot or torso height rewards would be maximized by toppling over).
    * ``jump`` -- launch reward in [0, 1] (see :class:`RewardSpec`).

    Every mode is monotone in its target quantity across the visited range.
    """
    torso_height = physics.torso_height()
    horizontal_velocity = physics.horizontal_velocity()
    upright = (1 + physics.torso_upright()) / 2
    pitch_rate = float(physics.named.data.qvel["rooty"])
    vertical_velocity = float(physics.named.data.qvel["rootz"])

    out = {}
    for name, spec in WALKER_REWARD_MODES.items():
        if spec.spin_rate is not None:
            rate = abs(spec.spin_rate)
            spin_dir = 1.0 if spec.spin_rate > 0 else -1.0
            out[name] = float(np.clip(spin_dir * pitch_rate / rate, -1.0, 1.0))
            continue
        if spec.leap is not None:
            h_lo, h_hi, vz_scale = spec.leap
            height_frac = float(np.clip((torso_height - h_lo) / (h_hi - h_lo), 0.0, 1.0))
            rise_frac = float(np.clip(vertical_velocity / vz_scale, 0.0, 1.0))
            out[name] = height_frac**2 * rise_frac
            continue
        height_term = dmc_rewards.tolerance(
            torso_height,
            bounds=spec.height_bounds,
            margin=spec.height_margin,
        )
        posture_reward = float((3 * height_term + upright) / 4)
        if spec.move_ramp is not None:
            v_zero, v_full, rest = spec.move_ramp
            v = horizontal_velocity
            wrong_side = (v >= 0) == (v_full < 0)
            if wrong_side:
                # wrong-direction side: 0 at/beyond v_zero, `rest` at rest
                if rest == 0.0 or v_zero == 0.0:
                    move_reward = 0.0
                else:
                    move_reward = rest * float(
                        np.clip((v_zero - v) / v_zero, 0.0, 1.0)
                    )
            else:
                # target side: ramp from `rest` at rest up to 1 at v_full
                move_reward = rest + (1.0 - rest) * float(
                    np.clip(v / v_full, 0.0, 1.0)
                )
            out[name] = float(height_term) * move_reward
            continue
        if spec.move_speed == 0:
            out[name] = posture_reward
            continue
        speed = abs(spec.move_speed)
        direction = 1.0 if spec.move_speed > 0 else -1.0
        margin = speed / 2 if spec.move_margin is None else spec.move_margin
        move_reward = dmc_rewards.tolerance(
            direction * horizontal_velocity,
            bounds=(speed, float("inf")),
            margin=margin,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        out[name] = float(posture_reward * (5 * move_reward + 1) / 6)
    return out


class _MultiRewardDMCR(DMC_Remastered_Env):
    """``DMC_Remastered_Env`` that scores every reward function per sub-step.

    The parent accumulates only the active task's reward across ``frame_skip``
    sub-steps; ``step`` is overridden to accumulate every mode over the same
    loop so all are on the same scale.
    """

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action = self._convert_action(action)
        totals = {name: 0.0 for name in WALKER_REWARD_MODES}
        truncated = False
        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            for name, value in walker_rewards(self._env.physics).items():
                totals[name] += value
            truncated = time_step.last()
            if truncated:
                break
        self.current_state = _flatten_obs(time_step.observation)
        return self._get_obs(time_step), totals, False, truncated, {}


class DMCRWalkerMeta(gym.Env):
    """k-shot meta-RL trial over randomized DMC Remastered walker bodies.

    A hard reset draws a dynamics seed from ``dynamics_seeds`` and recompiles
    the walker model, fixing the embodiment for the trial. The ``k-1``
    intra-trial resets are soft, so the agent gets ``k`` attempts at one body.

    ``reward_mode`` selects which mode is returned as ``reward``; every mode is
    scored and logged as a diagnostic. ``walk`` is the default because ``run``
    and ``walk`` share a stand term with speed gates at 8 vs 1 m/s:
    supervising ``run`` would make ``walk``/``stand`` nearly tautological,
    while supervising ``walk`` leaves ``run`` a harder OOD target.

    Args:
        dynamics_seeds: Pool of dynamics seeds to draw a body from on each hard
            reset. Typically one side of :func:`make_dynamics_seed_split`.
        reward_mode: Which of ``WALKER_REWARD_MODES`` drives ``reward``.
        k: Episodes per meta-trial.
        max_episode_steps: Agent steps per inner episode.
        frame_skip: Control steps per agent step. The reward is summed over the
            skip, so per-agent-step reward is bounded by ``frame_skip``.
        vary: Randomization axes passed to the DMCR model generator.
        seed: Seeds the body-sampling RNG. Pass a distinct value per parallel
            actor so they do not draw identical bodies in lockstep.
        render_dir: If set, write an mp4 per trial here (offscreen camera only;
            observations stay state-based). Enable on a single actor.
        render_trials: Stop after this many videos.

    Observation keys:
        ``observed``     float32 (25,)  24-D walker state + soft-reset flag.
        ``prev_action``  float32 (6,)   Action from *two* timesteps ago
            (``a_{t-2}`` at ``o_t``), as in ColoredGrid / Wind. IDM predicts
            ``a_t`` from ``(idm_t, idm_{t+1})``, so ``a_t`` or ``a_{t-1}`` in
            either observation would leak the label.
        ``episode_id``   int32   ()     Episode index within the trial, ``k`` at trial end.
        ``state_vel``    float32 (1,)   Torso horizontal velocity (diagnostic / RND target).
    """

    def __init__(
        self,
        dynamics_seeds: Sequence[int],
        reward_mode: str = "walk",
        k: int = 2,
        max_episode_steps: int = 500,
        frame_skip: int = 2,
        vary: Optional[Iterable[str]] = None,
        seed: int = 0,
        render_dir: Optional[str] = None,
        render_trials: int = 1,
    ):
        if reward_mode not in WALKER_REWARD_MODES:
            raise ValueError(
                f"reward_mode={reward_mode!r} not in {sorted(WALKER_REWARD_MODES)}"
            )
        self.dynamics_seeds = np.asarray(list(dynamics_seeds), dtype=np.int64)
        if self.dynamics_seeds.size == 0:
            raise ValueError("dynamics_seeds must be non-empty.")
        if (self.dynamics_seeds == 0).any():
            # get_model() short-circuits all dynamics randomization at seed 0.
            raise ValueError("dynamics_seed 0 disables randomization; use seeds >= 1.")

        self.reward_mode = reward_mode
        self.k = int(k)
        self.max_episode_steps = int(max_episode_steps)
        self.frame_skip = int(frame_skip)
        self.vary = list(DYNAMICS_VARY if vary is None else vary)
        self._rng = np.random.default_rng(seed)

        # Give dm_control's own time limit a couple of agent steps of slack so
        # our step counter is always what ends the episode. Otherwise float
        # rounding in control.Environment's step_limit could truncate at 499 or
        # 501 and make trial lengths ragged, which breaks the collection math.
        time_limit = (
            self.max_episode_steps * self.frame_skip + 2 * self.frame_skip
        ) * _CONTROL_TIMESTEP

        # Always build the `run` task: the XML is identical across builders and
        # rewards are computed here, so dynamics_seed -> body is stable across
        # reward modes.
        self._env = _MultiRewardDMCR(
            task_builder=partial(DYNAMICS_ENVS["walker"]["run"], time_limit=time_limit),
            dynamics_seed_generator=self._draw_dynamics_seed,
            visual_seed_generator=lambda: 0,
            from_pixels=False,
            frame_skip=self.frame_skip,
            vary=self.vary,
        )

        self.num_actions = int(self._env.action_space.shape[0])
        state_dim = int(self._env.observation_space.shape[0])
        self.observation_space = gym.spaces.Dict(
            {
                "observed": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(state_dim + 1,), dtype=np.float32
                ),
                "prev_action": gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
                ),
                "episode_id": gym.spaces.Box(
                    low=0, high=self.k, shape=(), dtype=np.int32
                ),
                "state_vel": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )

        self.current_trial = 0
        self.current_time = 0
        self.current_dynamics_seed = None
        # Two-slot lag buffer. obs["prev_action"] reads slot 0 (= a_{t-2});
        # after each step the buffer becomes [a_{t-1}, a_t]. See Observation
        # keys docstring for why a one-step lag would leak the IDM label.
        self._action_history = [
            np.zeros(self.num_actions, dtype=np.float32),
            np.zeros(self.num_actions, dtype=np.float32),
        ]
        self._episode_returns = {name: 0.0 for name in WALKER_REWARD_MODES}
        self._trial_returns = {name: 0.0 for name in WALKER_REWARD_MODES}
        self._episode_vel: List[float] = []

        self.render_dir = render_dir
        self.render_trials = int(render_trials)
        self._videos_saved = 0
        self._frames: List[np.ndarray] = []
        if self.render_dir is not None:
            os.makedirs(self.render_dir, exist_ok=True)

    # ── Video capture ─────────────────────────────────────────────────────

    @property
    def _recording(self) -> bool:
        return self.render_dir is not None and self._videos_saved < self.render_trials

    def _capture(self) -> None:
        if not self._recording:
            return
        self._frames.append(
            self._env.physics.render(height=240, width=320, camera_id=0)
        )

    def _save_video(self) -> None:
        """Write the buffered trial to mp4. Frames are one per agent step."""
        if not self._frames:
            return
        import imageio.v2 as imageio

        path = os.path.join(
            self.render_dir,
            f"{self.reward_mode}_trial{self._videos_saved:02d}"
            f"_seed{self.current_dynamics_seed}.mp4",
        )
        # 1 frame per agent step = frame_skip control steps of 0.025s each.
        fps = max(int(round(1.0 / (self.frame_skip * _CONTROL_TIMESTEP))), 1)
        imageio.mimwrite(path, self._frames, fps=fps, macro_block_size=1)
        self._videos_saved += 1
        self._frames = []

    # ── Body sampling ─────────────────────────────────────────────────────

    def _draw_dynamics_seed(self) -> int:
        seed = int(self.dynamics_seeds[self._rng.integers(self.dynamics_seeds.size)])
        self.current_dynamics_seed = seed
        return seed

    # ── Observation ───────────────────────────────────────────────────────

    def _build_obs(self, state: np.ndarray, soft_reset: bool) -> dict:
        return {
            "observed": np.concatenate(
                [np.asarray(state, dtype=np.float32), [np.float32(soft_reset)]]
            ).astype(np.float32),
            "prev_action": self._action_history[0].copy(),
            "episode_id": np.int32(self.current_trial),
            "state_vel": np.array(
                [self._env.physics.horizontal_velocity()], dtype=np.float32
            ),
        }

    # ── Gym API ───────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        # DMC_Remastered_Env.reset() asserts seed is None (seeding is a
        # constructor concern there), but AMAGOEnv.reset() always forwards a
        # seed kwarg. Absorb it here and reseed our own body sampler instead.
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.current_trial = 0
        self.current_time = 0
        self._action_history = [
            np.zeros(self.num_actions, dtype=np.float32),
            np.zeros(self.num_actions, dtype=np.float32),
        ]
        self._episode_returns = {name: 0.0 for name in WALKER_REWARD_MODES}
        self._trial_returns = {name: 0.0 for name in WALKER_REWARD_MODES}
        self._episode_vel = []

        state, _ = self._env.reset(soft=False)
        # Drop any partial buffer: a trial cut short mid-recording would
        # otherwise be spliced onto the front of the next one.
        self._frames = []
        self._capture()
        return self._build_obs(state, soft_reset=True), {}

    def step(self, action):
        state, rewards, _, inner_truncated, _ = self._env.step(action)
        self.current_time += 1
        for name, value in rewards.items():
            self._episode_returns[name] += value
        self._episode_vel.append(float(self._env.physics.horizontal_velocity()))

        info = {}
        soft_reset = self.current_time >= self.max_episode_steps or inner_truncated
        if soft_reset:
            self.current_trial += 1
            self.current_time = 0
            for name, ep_return in self._episode_returns.items():
                self._trial_returns[name] += ep_return
                info[
                    f"{AMAGO_ENV_LOG_PREFIX} Episode {self.current_trial} "
                    f"DMC Score ({name})"
                ] = self._dmc_score(ep_return)
            # Ground-truth locomotion readout. Reward scores cannot separate
            # standing from backward travel (both zero out the forward-only
            # move terms), and the periodic checkerboard floor defeats
            # optical-flow estimates off the rendered video, so log signed
            # velocity directly.
            vel = np.asarray(self._episode_vel, dtype=np.float64)
            info[
                f"{AMAGO_ENV_LOG_PREFIX} Episode {self.current_trial} Mean Vel"
            ] = float(vel.mean())
            info[
                f"{AMAGO_ENV_LOG_PREFIX} Episode {self.current_trial} Frac Vel Backward"
            ] = float((vel < -0.1).mean())
            self._episode_vel = []
            self._episode_returns = {name: 0.0 for name in WALKER_REWARD_MODES}
            # Soft reset keeps the compiled body; only the episode restarts.
            state, _ = self._env.reset(soft=True)

        self._capture()

        terminated = truncated = self.current_trial >= self.k
        if terminated:
            for name, trial_return in self._trial_returns.items():
                info[f"{AMAGO_ENV_LOG_PREFIX} Trial Return ({name})"] = trial_return
                info[f"{AMAGO_ENV_LOG_PREFIX} Trial Mean DMC Score ({name})"] = (
                    self._dmc_score(trial_return / self.k)
                )
            self._save_video()

        # Build obs BEFORE shifting the history so slot 0 is still a_{t-2}.
        obs = self._build_obs(state, soft_reset=soft_reset)
        action_np = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._action_history = [self._action_history[1].copy(), action_np.copy()]
        return obs, rewards[self.reward_mode], terminated, truncated, info

    def _dmc_score(self, episode_return: float) -> float:
        """Rescale an episode return onto the canonical DMC 0-1000 scale."""
        return (
            DMC_SCORE_SCALE
            * episode_return
            / (self.max_episode_steps * self.frame_skip)
        )


# ── Dynamics-seed splits ──────────────────────────────────────────────────


@dataclass
class DynamicsSeedSplit:
    """Disjoint train / test pools of DMCR dynamics seeds."""

    train: List[int]
    test: List[int]


#: Largest supported train pool. Mirrors ``reward_pool._MAX_TRAIN`` so that the
#: train prefix for a smaller ``n_train`` is a subset of the larger ones.
_MAX_TRAIN = 10_000

#: Seeds are drawn from [1, _SEED_CEILING). 0 is excluded because
#: ``walker.get_model`` treats it as "no dynamics randomization".
_SEED_CEILING = 1_000_000


def make_dynamics_seed_split(
    n_train: int,
    n_test: int = 1_000,
    seed: int = 0,
) -> DynamicsSeedSplit:
    """Build genuinely disjoint train / test dynamics-seed pools.

    ``dmc_remastered.benchmarks.dynamics_generalization`` is not usable for
    this: its test generator is ``uniform_seed_generator(1, 1_000_000)``, which
    covers the train range too, so "held-out" bodies leak into training.

    Instead draw ``_MAX_TRAIN + n_test`` distinct seeds once; train is a prefix
    and test is the final ``n_test``. As in
    :func:`metabfm.env.reward_pool.make_reward_split`, the test set is fixed
    across the ``n_train`` sweep and train pools nest.
    """
    if not 0 <= n_train <= _MAX_TRAIN:
        raise ValueError(f"n_train={n_train} must be in [0, {_MAX_TRAIN}]")
    rng = np.random.default_rng(seed)
    pool = rng.choice(
        np.arange(1, _SEED_CEILING), size=_MAX_TRAIN + n_test, replace=False
    )
    return DynamicsSeedSplit(
        train=[int(s) for s in pool[:n_train]],
        test=[int(s) for s in pool[_MAX_TRAIN:]],
    )
