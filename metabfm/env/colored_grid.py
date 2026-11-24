import os
import warnings
from uuid import uuid4
from typing import Optional, List

warnings.filterwarnings("ignore", message="non-integer arguments to randrange")

import gymnasium as gym
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


from amago.envs import AMAGO_ENV_LOG_PREFIX


def _sparsify_reward_grid(grid: np.ndarray) -> np.ndarray:
    """Zero all cells then place the max in the top-left corner and the min
    in the center of the grid. The top-left corner is the hardest cell to
    stumble into from a uniformly-sampled start, which makes the task a
    harder exploration problem while remaining fully compatible with the
    oracle return (which is computed directly from argmax(reward_grid))."""
    out = np.zeros_like(grid)
    flat = grid.ravel()
    max_val = float(flat.max())
    min_val = float(flat.min())
    n_rows, n_cols = grid.shape
    out[0, 0] = max_val
    out[n_rows // 2, n_cols // 2] = min_val
    return out


class ColoredGrid(gym.Env):
    def __init__(
        self,
        n: int,
        k: int,
        max_ep_steps: int = 50,
        randomize_tilt: bool = False,
        max_tilt: float = 0.35,
        seed: Optional[int] = None,
        cell_perms: Optional[List[np.ndarray]] = None,
        reward_grids: Optional[List[np.ndarray]] = None,
        render_dir: Optional[str] = None,
        fps: int = 15,
        cell_px: int = 64,
        color_seed: int = 42,
        fixed_start: bool = False,
        sparse_mode: bool = False,
    ):
        self.max_tilt = max_tilt
        self.observation_space = gym.spaces.Dict(
            {
                "observed": gym.spaces.Box(0, n, shape=(3,), dtype=np.float32),
                "actual": gym.spaces.Box(0, n, shape=(2,), dtype=np.float32),
                "obs_colors": gym.spaces.Box(0, 255, shape=(n, n, 3), dtype=np.uint8),
                "state_colors": gym.spaces.Box(0, 255, shape=(n, n, 3), dtype=np.uint8),
                "reward_grid": gym.spaces.Box(
                    -10.0, 10.0, shape=(n, n), dtype=np.float32
                ),
                "episode_id": gym.spaces.Box(0, k, shape=(), dtype=np.int32),
                "prev_action": gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
                "prev_reward": gym.spaces.Box(
                    -10.0, 10.0, shape=(1,), dtype=np.float32
                ),
                "prev_terminal": gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "tilt": gym.spaces.Box(
                    -self.max_tilt, self.max_tilt, shape=(2,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.n = n
        self.k = k
        self.max_ep_steps = max_ep_steps
        self.randomize_tilt = randomize_tilt
        self.rng = np.random.default_rng(seed)
        if cell_perms is not None:
            self._cell_perms = cell_perms
        else:
            fixed_rng = np.random.default_rng(0)
            self._cell_perms = [fixed_rng.permutation(n * n)]
        if reward_grids is not None:
            self._reward_grids = reward_grids
        else:
            fixed_rng = np.random.default_rng(0)
            grid = fixed_rng.normal(0.0, 3.0, size=(n, n)).astype(np.float32)
            self._reward_grids = [np.clip(grid, -10.0, 10.0)]
        self.sparse_mode = sparse_mode
        if sparse_mode:
            self._reward_grids = [
                _sparsify_reward_grid(g).astype(np.float32) for g in self._reward_grids
            ]
        self.render_dir = render_dir
        self.fps = fps
        self.cell_px = cell_px
        self.color_seed = color_seed
        self.fixed_start = fixed_start
        self.video_count = 0
        self.font = self._load_font(int(self.cell_px * 0.28))
        self.font_small = self._load_font(int(self.cell_px * 0.2))
        self._color_grid = self._make_color_grid(n)
        if self.render_dir is not None:
            os.makedirs(self.render_dir, exist_ok=True)
        self.reset()

    def build_grid(self):
        cell_perm = self._cell_perms[
            int(self.rng.integers(len(self._cell_perms)))
        ].copy()
        idx = int(self.rng.integers(len(self._reward_grids)))
        reward_grid = self._reward_grids[idx].copy()
        color_grid = self._color_grid.copy()
        inv_perm = np.argsort(cell_perm)
        obs_colors = color_grid.reshape(self.n * self.n, 3)[inv_perm].reshape(
            self.n, self.n, 3
        )
        return {
            "cell_perm": cell_perm,
            "reward_grid": reward_grid,
            "color_grid": color_grid,
            "state_colors": color_grid,
            "obs_colors": obs_colors,
        }

    def render_obs(self, pos: np.ndarray):
        pos = np.clip(pos, 0.0, self.n - 1e-6)
        x_idx = int(np.floor(pos[0]))
        y_idx = int(np.floor(pos[1]))
        x_frac = pos[0] - x_idx
        y_frac = pos[1] - y_idx
        flat_idx = y_idx * self.n + x_idx
        mapped = self.grid["cell_perm"][flat_idx]
        mapped_x = mapped % self.n
        mapped_y = mapped // self.n
        reset_flag = 1.0 if self.just_reset else 0.0
        observed = np.array(
            [
                mapped_x + x_frac,
                mapped_y + y_frac,
                reset_flag,
            ],
            dtype=np.float32,
        )
        actual = np.array(pos, dtype=np.float32)
        return {
            "observed": observed,
            "actual": actual,
            "obs_colors": self.grid["obs_colors"],
            "state_colors": self.grid["state_colors"],
            "reward_grid": self.grid["reward_grid"],
            "episode_id": np.int32(self.episode),
            "prev_action": self._action_history[0].copy(),
            "prev_reward": np.array([self._reward_history[0]], dtype=np.float32),
            "prev_terminal": np.array(
                [float(self._terminal_history[0])], dtype=np.float32
            ),
            "tilt": self.tilt.astype(np.float32),
        }

    def reset(self, *args, **kwargs):
        if self.fixed_start:
            self.pos = np.zeros(2, dtype=np.float64)
        else:
            self.pos = self.rng.random((2,)) * self.n
        self.ep_start_pos = self.pos.copy()
        self.episode = 0
        self.time = 0
        self.grid = self.build_grid()
        self.episode_return = 0.0
        self._total_return = 0.0
        self._total_oracle_return = 0.0
        self._total_top10_oracle_return = 0.0
        if self.randomize_tilt:
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            magnitude = self.rng.uniform(0.0, self.max_tilt)
            self.tilt = magnitude * np.array(
                [np.cos(angle), np.sin(angle)], dtype=np.float32
            )
        else:
            self.tilt = np.zeros(2, dtype=np.float32)
        if self.render_dir is not None:
            self.frames = []
            self.prev_actual_pos = None
            self.prev_observed_pos = None
        self._action_history = [
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        ]
        self._reward_history = [0.0, 0.0]
        self._terminal_history = [False, False]
        self.just_reset = True
        self.flash_border = False
        obs = self.render_obs(self.pos)
        if self.render_dir is not None:
            self._record_frame(obs["actual"], None, obs["observed"][:2], None)
            self.prev_actual_pos = obs["actual"].copy()
            self.prev_observed_pos = obs["observed"][:2].copy()
        return obs, {}

    def step(self, action: np.ndarray):
        self.time += 1
        info = {}
        if self.time >= self.max_ep_steps:
            prev_start_pos = self.ep_start_pos.copy()
            reward_grid = self.grid["reward_grid"]
            target_flat_idx = int(np.argmax(reward_grid))
            max_reward = float(reward_grid.flat[target_flat_idx])
            target_x = target_flat_idx % self.n
            target_y = target_flat_idx // self.n

            def axis_dist(start: float, cell_idx: int) -> float:
                if cell_idx <= start < cell_idx + 1.0:
                    return 0.0
                if start < cell_idx:
                    return cell_idx - start
                return start - (cell_idx + 1.0)

            top10_reward = float(
                np.sort(reward_grid.ravel())[-min(10, reward_grid.size) :].mean()
            )

            dx = axis_dist(float(prev_start_pos[0]), target_x)
            dy = axis_dist(float(prev_start_pos[1]), target_y)
            steps_to_reach = int(np.ceil(max(dx, dy)))
            remaining_steps = max(0, self.max_ep_steps - steps_to_reach)
            oracle_return = float(remaining_steps) * max_reward
            top10_oracle_return = float(remaining_steps) * top10_reward
            self.episode += 1
            self.time = 0
            if self.fixed_start:
                self.pos = np.zeros(2, dtype=np.float64)
            else:
                self.pos = self.rng.random((2,)) * self.n
            self.ep_start_pos = self.pos.copy()
            self.just_reset = True
            self.flash_border = True
            info[f"{AMAGO_ENV_LOG_PREFIX} Episode {self.episode} Return"] = (
                self.episode_return
            )
            info[f"{AMAGO_ENV_LOG_PREFIX} Episode {self.episode} Oracle Return"] = (
                oracle_return
            )
            info[
                f"{AMAGO_ENV_LOG_PREFIX} Episode {self.episode} Oracle Normalized Return"
            ] = (self.episode_return / oracle_return)
            info[
                f"{AMAGO_ENV_LOG_PREFIX} Episode {self.episode} Top10 Oracle Normalized Return"
            ] = (
                self.episode_return / top10_oracle_return
                if top10_oracle_return > 0
                else 0.0
            )
            self._total_return += self.episode_return
            self._total_oracle_return += oracle_return
            self._total_top10_oracle_return += top10_oracle_return
            self.episode_return = 0.0
        else:
            self.pos = np.clip(self.pos + action + self.tilt, 0.0, self.n - 1e-6)
        obs = self.render_obs(self.pos)
        reward = self.compute_reward(self.pos)
        is_boundary = self.just_reset
        self._action_history = [
            self._action_history[1].copy(),
            np.asarray(action, dtype=np.float32).copy(),
        ]
        self._reward_history = [self._reward_history[1], float(reward)]
        self._terminal_history = [self._terminal_history[1], is_boundary]
        self.episode_return += reward
        terminated = self.episode >= self.k
        if terminated and self._total_oracle_return > 0:
            info[f"{AMAGO_ENV_LOG_PREFIX} Oracle Normalized Return"] = (
                self._total_return / self._total_oracle_return
            )
        if terminated and self._total_top10_oracle_return > 0:
            info[f"{AMAGO_ENV_LOG_PREFIX} Top10 Oracle Normalized Return"] = (
                self._total_return / self._total_top10_oracle_return
            )
        truncated = False
        if self.render_dir is not None:
            if self.just_reset:
                self._record_frame(obs["actual"], None, obs["observed"][:2], None)
            else:
                self._record_frame(
                    obs["actual"],
                    self.prev_actual_pos,
                    obs["observed"][:2],
                    self.prev_observed_pos,
                )
            self.prev_actual_pos = obs["actual"].copy()
            self.prev_observed_pos = obs["observed"][:2].copy()
            if terminated:
                self._save_video()
        if self.just_reset:
            self.just_reset = False
        return obs, reward, terminated, truncated, info

    def compute_reward(self, pos: np.ndarray):
        pos = np.clip(pos, 0.0, self.n - 1e-6)
        x_idx = int(np.floor(pos[0]))
        y_idx = int(np.floor(pos[1]))
        return float(self.grid["reward_grid"][y_idx, x_idx])

    def _make_color_grid(self, n: int) -> np.ndarray:
        # Use fixed seed for consistent colors across all grids of the same size
        color_rng = np.random.default_rng(self.color_seed)
        num_cells = n * n
        # Evenly space hues around the color wheel for maximum distinction
        base_hues = np.linspace(0, 1, num_cells, endpoint=False)
        # Shuffle hues so adjacent cells don't have similar colors
        color_rng.shuffle(base_hues)
        # Add small jitter to make it less mechanical
        h = (base_hues + color_rng.uniform(-0.02, 0.02, size=num_cells)) % 1.0
        h = h.reshape(n, n)
        # Vary saturation and value for additional distinction
        s = color_rng.uniform(0.55, 0.85, size=(n, n))
        v = color_rng.uniform(0.75, 0.95, size=(n, n))
        rgb = self._hsv_to_rgb(h, s, v)
        return (rgb * 255).astype(np.uint8)

    def _hsv_to_rgb(self, h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
        i = np.floor(h * 6).astype(int)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i = i % 6
        r = np.zeros_like(h)
        g = np.zeros_like(h)
        b = np.zeros_like(h)
        mask = i == 0
        r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
        mask = i == 1
        r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
        mask = i == 2
        r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
        mask = i == 3
        r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
        mask = i == 4
        r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
        mask = i == 5
        r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]
        return np.stack([r, g, b], axis=-1)

    def _record_frame(
        self,
        actual_pos: np.ndarray,
        prev_actual_pos: Optional[np.ndarray],
        observed_pos: np.ndarray,
        prev_observed_pos: Optional[np.ndarray],
    ):
        frame = self._render_frame(
            actual_pos, prev_actual_pos, observed_pos, prev_observed_pos
        )
        self.frames.append(frame)
        self.flash_border = False

    def _render_frame(
        self,
        actual_pos: np.ndarray,
        prev_actual_pos: Optional[np.ndarray],
        observed_pos: np.ndarray,
        prev_observed_pos: Optional[np.ndarray],
    ) -> np.ndarray:
        panel_pad = max(12, int(self.cell_px * 0.3))
        panel_gap = int(self.cell_px * 0.6)
        size = self.n * self.cell_px
        tilt_mag = float(np.linalg.norm(self.tilt))
        if tilt_mag > 0:
            tilt_dir = self.tilt / tilt_mag
        else:
            tilt_dir = np.zeros(2, dtype=np.float32)
        skew = np.zeros(2, dtype=np.float32)
        panel_width = int(size + panel_pad * 2 + abs(skew[0]) * (self.n - 1))
        panel_height = int(size + panel_pad * 2 + abs(skew[1]) * (self.n - 1))
        title_pad = int(self.cell_px * 0.6)
        width = int(panel_width * 2 + panel_gap)
        status_pad = int(self.cell_px * 0.45)
        height = int(panel_height + status_pad + title_pad)
        img = Image.new("RGBA", (width, height), (235, 235, 235, 255))
        draw = ImageDraw.Draw(img, "RGBA")
        top_bg = np.array([246, 246, 246], dtype=np.float32)
        bot_bg = np.array([228, 228, 228], dtype=np.float32)
        for y in range(height):
            t = y / max(1, height - 1)
            color = tuple((top_bg * (1 - t) + bot_bg * t).astype(np.uint8)) + (255,)
            draw.line([(0, y), (width, y)], fill=color)
        grid_center = np.array([self.n / 2.0, self.n / 2.0], dtype=np.float32)

        # Colors and rewards belong to state cells; the observation panel shows
        # each at its permuted location.
        inv_perm = np.argsort(self.grid["cell_perm"])

        def draw_grid(offset_x: float, offset_y: float, is_state_space: bool):
            border_top_left = (offset_x + panel_pad, offset_y + panel_pad)
            border_top_right = (offset_x + panel_pad + size, offset_y + panel_pad)
            border_bottom_left = (
                offset_x + panel_pad + skew[0] * (self.n - 1),
                offset_y + panel_pad + size + skew[1] * (self.n - 1),
            )
            border_bottom_right = (
                offset_x + panel_pad + size + skew[0] * (self.n - 1),
                offset_y + panel_pad + size + skew[1] * (self.n - 1),
            )
            shadow_offset = (2, 2)
            shadow_poly = [
                (
                    border_top_left[0] + shadow_offset[0],
                    border_top_left[1] + shadow_offset[1],
                ),
                (
                    border_top_right[0] + shadow_offset[0],
                    border_top_right[1] + shadow_offset[1],
                ),
                (
                    border_bottom_right[0] + shadow_offset[0],
                    border_bottom_right[1] + shadow_offset[1],
                ),
                (
                    border_bottom_left[0] + shadow_offset[0],
                    border_bottom_left[1] + shadow_offset[1],
                ),
            ]
            draw.polygon(shadow_poly, fill=(0, 0, 0, 25))

            for y in range(self.n):
                for x in range(self.n):
                    flat_idx = y * self.n + x

                    if is_state_space:
                        color_y, color_x = y, x
                        reward_y, reward_x = y, x
                    else:
                        # Colors: apply inverse perm (show color of state that maps here)
                        state_flat = inv_perm[flat_idx]
                        color_x = state_flat % self.n
                        color_y = state_flat // self.n
                        # Rewards: show reward of the state that maps to this obs cell
                        reward_x, reward_y = color_x, color_y

                    base_x = offset_x + panel_pad + x * self.cell_px + skew[0] * y
                    base_y = offset_y + panel_pad + y * self.cell_px + skew[1] * y
                    poly = [
                        (base_x, base_y),
                        (base_x + self.cell_px, base_y),
                        (base_x + self.cell_px, base_y + self.cell_px),
                        (base_x, base_y + self.cell_px),
                    ]
                    color = self.grid["color_grid"][color_y, color_x].astype(np.float32)
                    color = tuple(color.astype(np.uint8)) + (255,)
                    draw.polygon(poly, fill=color)
                    draw.polygon(poly, outline=(0, 0, 0, 70), width=1)
                    reward = float(self.grid["reward_grid"][reward_y, reward_x])
                    text = f"{reward:+.1f}"
                    text_color = (0, 0, 0, 255)
                    bbox = draw.textbbox((0, 0), text, font=self.font)
                    tx = base_x + (self.cell_px - (bbox[2] - bbox[0])) / 2
                    ty = base_y + (self.cell_px - (bbox[3] - bbox[1])) / 2
                    draw.text((tx, ty), text, fill=text_color, font=self.font)

            if tilt_mag > 0:
                perp = np.array([-tilt_dir[1], tilt_dir[0]], dtype=np.float32)
                spacing = 0.95
                offsets = np.arange(-self.n, self.n + spacing, spacing)
                arrow_len = self.cell_px * 0.16
                for d in offsets:
                    p0, p1 = self._clip_line_to_box(
                        grid_center + tilt_dir * d, perp, 0.0, self.n
                    )
                    if p0 is None:
                        continue
                    x0 = offset_x + panel_pad + p0[0] * self.cell_px + skew[0] * p0[1]
                    y0 = offset_y + panel_pad + p0[1] * self.cell_px + skew[1] * p0[1]
                    x1 = offset_x + panel_pad + p1[0] * self.cell_px + skew[0] * p1[1]
                    y1 = offset_y + panel_pad + p1[1] * self.cell_px + skew[1] * p1[1]
                    self._draw_dashed_line(
                        draw,
                        (x0, y0),
                        (x1, y1),
                        color=(0, 0, 0, 35),
                        dash_len=6,
                        gap_len=6,
                        width=1,
                    )
                    step = self.cell_px * 0.7
                    length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                    if length == 0:
                        continue
                    ux = (x1 - x0) / length
                    uy = (y1 - y0) / length
                    dist = step / 2
                    while dist < length:
                        cx = x0 + ux * dist
                        cy = y0 + uy * dist
                        end = (
                            cx + tilt_dir[0] * arrow_len,
                            cy + tilt_dir[1] * arrow_len,
                        )
                        if (
                            offset_x + panel_pad <= cx <= offset_x + panel_pad + size
                            and offset_y + panel_pad
                            <= cy
                            <= offset_y + panel_pad + size
                        ):
                            self._draw_arrow(
                                draw, (cx, cy), end, color=(0, 0, 0, 140), width=1
                            )
                        dist += step

            border_color = (220, 40, 40, 255) if self.flash_border else (0, 0, 0, 255)
            border_width = 5 if self.flash_border else 3
            draw.polygon(
                [
                    border_top_left,
                    border_top_right,
                    border_bottom_right,
                    border_bottom_left,
                ],
                outline=border_color,
                width=border_width,
            )

        left_offset_x = 0
        right_offset_x = panel_width + panel_gap
        panel_offset_y = title_pad
        draw_grid(left_offset_x, panel_offset_y, is_state_space=True)
        draw_grid(right_offset_x, panel_offset_y, is_state_space=False)

        gap_x = panel_width + panel_gap / 2
        draw.line(
            [(gap_x, panel_offset_y), (gap_x, panel_offset_y + panel_height)],
            fill=(0, 0, 0, 110),
            width=2,
        )

        def draw_agent(
            offset_x: float,
            offset_y: float,
            pos: np.ndarray,
            prev_pos: Optional[np.ndarray],
        ):
            if prev_pos is not None:
                prev_px = (
                    offset_x
                    + panel_pad
                    + prev_pos[0] * self.cell_px
                    + skew[0] * prev_pos[1]
                )
                prev_py = (
                    offset_y
                    + panel_pad
                    + prev_pos[1] * self.cell_px
                    + skew[1] * prev_pos[1]
                )
                curr_px = (
                    offset_x + panel_pad + pos[0] * self.cell_px + skew[0] * pos[1]
                )
                curr_py = (
                    offset_y + panel_pad + pos[1] * self.cell_px + skew[1] * pos[1]
                )
                self._draw_arrow(
                    draw,
                    (prev_px, prev_py),
                    (curr_px, curr_py),
                    color=(0, 0, 0, 255),
                    width=3,
                )

            curr_px = offset_x + panel_pad + pos[0] * self.cell_px + skew[0] * pos[1]
            curr_py = offset_y + panel_pad + pos[1] * self.cell_px + skew[1] * pos[1]
            curr_r = self.cell_px * 0.2
            draw.ellipse(
                [
                    curr_px - curr_r,
                    curr_py - curr_r,
                    curr_px + curr_r,
                    curr_py + curr_r,
                ],
                fill=(255, 255, 255, 255),
                outline=(0, 0, 0, 255),
                width=2,
            )

        draw_agent(left_offset_x, panel_offset_y, actual_pos, prev_actual_pos)
        draw_agent(right_offset_x, panel_offset_y, observed_pos, prev_observed_pos)

        title_left = "State Space"
        title_right = "Observation Space"
        left_bbox = draw.textbbox((0, 0), title_left, font=self.font_small)
        right_bbox = draw.textbbox((0, 0), title_right, font=self.font_small)
        skew_center = skew[0] * (self.n - 1) / 2
        left_center_x = left_offset_x + panel_pad + size / 2 + skew_center
        right_center_x = right_offset_x + panel_pad + size / 2 + skew_center
        left_x = left_center_x - (left_bbox[2] - left_bbox[0]) / 2
        right_x = right_center_x - (right_bbox[2] - right_bbox[0]) / 2
        title_y = 6
        title_pad_x = 10
        title_pad_y = 5
        for x, text, bbox in (
            (left_x, title_left, left_bbox),
            (right_x, title_right, right_bbox),
        ):
            rect = [
                x - title_pad_x,
                title_y - title_pad_y,
                x + (bbox[2] - bbox[0]) + title_pad_x,
                title_y + (bbox[3] - bbox[1]) + title_pad_y,
            ]
            if hasattr(draw, "rounded_rectangle"):
                draw.rounded_rectangle(
                    rect,
                    radius=8,
                    fill=(255, 255, 255, 220),
                    outline=(0, 0, 0, 30),
                    width=1,
                )
            else:
                draw.rectangle(
                    rect, fill=(255, 255, 255, 220), outline=(0, 0, 0, 30), width=1
                )
            draw.text((x, title_y), text, fill=(0, 0, 0, 255), font=self.font_small)

        status = f"episode {self.episode}  t {self.time}"
        status = f"Episode {self.episode}, Step {self.time}"
        draw.text(
            (panel_pad, height - status_pad + 4),
            status,
            fill=(0, 0, 0, 255),
            font=self.font_small,
        )

        return np.array(img.convert("RGB"))

    def _draw_arrow(
        self, draw: ImageDraw.ImageDraw, start, end, color=(0, 0, 0, 255), width=2
    ):
        draw.line([start, end], fill=color, width=width)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            return
        ux = dx / length
        uy = dy / length
        head_len = max(6, width * 2)
        left = (
            end[0] - ux * head_len - uy * head_len * 0.5,
            end[1] - uy * head_len + ux * head_len * 0.5,
        )
        right = (
            end[0] - ux * head_len + uy * head_len * 0.5,
            end[1] - uy * head_len - ux * head_len * 0.5,
        )
        draw.polygon([end, left, right], fill=color)

    def _draw_dashed_line(
        self,
        draw: ImageDraw.ImageDraw,
        start,
        end,
        color=(0, 0, 0, 255),
        dash_len: int = 6,
        gap_len: int = 6,
        width: int = 1,
    ):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            return
        ux = dx / length
        uy = dy / length
        step = dash_len + gap_len
        dist = 0.0
        while dist < length:
            seg_start = dist
            seg_end = min(dist + dash_len, length)
            x0 = start[0] + ux * seg_start
            y0 = start[1] + uy * seg_start
            x1 = start[0] + ux * seg_end
            y1 = start[1] + uy * seg_end
            draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
            dist += step

    def _clip_line_to_box(
        self, point: np.ndarray, direction: np.ndarray, min_val: float, max_val: float
    ):
        t_min = -np.inf
        t_max = np.inf
        for i in range(2):
            p = direction[i]
            if abs(p) < 1e-6:
                if point[i] < min_val or point[i] > max_val:
                    return None, None
                continue
            t0 = (min_val - point[i]) / p
            t1 = (max_val - point[i]) / p
            if t0 > t1:
                t0, t1 = t1, t0
            t_min = max(t_min, t0)
            t_max = min(t_max, t1)
            if t_min > t_max:
                return None, None
        p0 = point + direction * t_min
        p1 = point + direction * t_max
        return p0, p1

    def _save_video(self):
        if not self.frames:
            return
        suffix = uuid4().hex[:8]
        filename = f"episode_{self.video_count:04d}_{suffix}.mp4"
        path = os.path.join(self.render_dir, filename)
        padded = []
        for frame in self.frames:
            height, width = frame.shape[:2]
            pad_h = height % 2
            pad_w = width % 2
            if pad_h or pad_w:
                frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            padded.append(frame)
        import imageio_ffmpeg

        h, w = padded[0].shape[:2]
        gen = imageio_ffmpeg.write_frames(
            path,
            (w, h),
            fps=self.fps,
            macro_block_size=1,
        )
        gen.send(None)
        for frame in padded:
            gen.send(frame.tobytes())
        gen.close()
        self.video_count += 1
        self.frames = []

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size=size)
        except OSError:
            return ImageFont.load_default()
