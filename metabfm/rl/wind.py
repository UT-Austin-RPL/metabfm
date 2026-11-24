"""Dict-observation wrapper for amago's Wind 2-D point-robot env.

``WindEnv`` returns a flat ``(3,)`` Box ``[x, y, soft_reset]`` and already
handles k-episode meta-trial state (``current_trial``, soft-reset flag,
per-trial metrics), so this module only repackages observations into the
``Dict`` interface the MetaBFM / AMAGO stack consumes.
"""

from typing import List

import gymnasium as gym
import numpy as np

from amago.envs.builtin.wind import WindEnv


class WindDictObs(gym.Wrapper):
    """Convert ``WindEnv``'s flat ``(3,)`` Box obs into the Dict interface
    expected by the MetaBFM / AMAGO stack.

    Keys
    ----
    observed     float32 (3,)         Raw ``[x, y, soft_reset]`` from WindEnv.
    prev_action  float32 (act_dim,)   Action from two timesteps ago, rescaled
                                      from ``WindEnv``'s ``[-0.1, 0.1]`` to
                                      the policy ``[-1, 1]`` range
                                      (PolicyTstepEncoder applies no InputNorm
                                      and expects unit-scale features).
    episode_id   int32   ()           Inner episode index within the meta-trial
                                      (== ``k_episodes`` at meta-trial truncation).
    state_pos    float32 (2,)         Agent ``(x, y)`` position; the RND target.
    """

    def __init__(self, env: WindEnv):
        super().__init__(env)
        self.k_episodes = env.k_episodes
        act_space = env.action_space
        self.num_actions = act_space.shape[0]
        # Inner action range, used to map stored prev_actions back to [-1, 1].
        self._inner_action_low = act_space.low.astype(np.float32)
        self._inner_action_high = act_space.high.astype(np.float32)
        self._inner_action_center = (
            self._inner_action_high + self._inner_action_low
        ) / 2.0
        self._inner_action_half_range = (
            self._inner_action_high - self._inner_action_low
        ) / 2.0
        flat_space = env.observation_space

        self.observation_space = gym.spaces.Dict(
            {
                "observed": gym.spaces.Box(
                    low=flat_space.low,
                    high=flat_space.high,
                    shape=flat_space.shape,
                    dtype=np.float32,
                ),
                "prev_action": gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.num_actions,),
                    dtype=np.float32,
                ),
                "episode_id": gym.spaces.Box(
                    low=0,
                    high=self.k_episodes,
                    shape=(),
                    dtype=np.int32,
                ),
                "state_pos": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )

        self._action_history: List[np.ndarray] = []
        self._reset_history()

    def _reset_history(self):
        z = np.zeros(self.num_actions, dtype=np.float32)
        self._action_history = [z.copy(), z.copy()]

    def _episode_id(self, trial_done: bool) -> np.int32:
        if trial_done:
            return np.int32(self.k_episodes)
        return np.int32(self.env.current_trial)

    def _build_dict(self, raw_obs: np.ndarray, trial_done: bool) -> dict:
        return {
            "observed": raw_obs.astype(np.float32),
            "prev_action": self._action_history[0].copy(),
            "episode_id": self._episode_id(trial_done),
            "state_pos": raw_obs[:2].astype(np.float32),
        }

    def reset(self, *, seed=None, options=None):
        raw_obs, info = self.env.reset(seed=seed, options=options)
        self._reset_history()
        return self._build_dict(raw_obs, trial_done=False), info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        trial_done = terminated or truncated
        dict_obs = self._build_dict(raw_obs, trial_done)

        # AMAGOEnv's ContinuousActionWrapper sits between AMAGOEnv and this
        # wrapper, so `action` arrives in WindEnv's native [-0.1, 0.1] range.
        # Rescale back to [-1, 1] to match the declared prev_action bounds.
        action_np = np.asarray(action, dtype=np.float32)
        norm_action = (
            (action_np - self._inner_action_center) / self._inner_action_half_range
        ).astype(np.float32)
        self._action_history = [self._action_history[1].copy(), norm_action.copy()]

        return dict_obs, reward, terminated, truncated, info
