"""Gymnasium MuJoCo hfield environment for a simple skid-steer vehicle."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    gym = None
    spaces = None
    _GYM_ERROR = exc

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    mujoco = None
    _MUJOCO_ERROR = exc

from terrain_interface import HeightmapTerrain


def _require_deps() -> None:
    if gym is None or mujoco is None:
        raise ImportError("sim_env requires gymnasium and mujoco")


class TerrainSkidSteerEnv(gym.Env if gym else object):
    """Four-wheel torque-controlled vehicle with local terrain observations."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, terrain: HeightmapTerrain, render_mode: Optional[str] = None,
                 patch_size: int = 9, max_steps: int = 500) -> None:
        _require_deps()
        if patch_size < 3 or patch_size % 2 == 0:
            raise ValueError("patch_size must be odd and >= 3")
        self.terrain, self.render_mode = terrain, render_mode
        self.patch_size, self.max_steps = patch_size, max_steps
        self.elevation = terrain.elevation()
        self.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "patch": spaces.Box(-np.inf, np.inf, (1, patch_size, patch_size), dtype=np.float32),
            "orientation": spaces.Box(-1, 1, (4,), dtype=np.float32),
            "velocity": spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
        })
        self.model = mujoco.MjModel.from_xml_string(self._xml())
        normalized = self.elevation - float(self.elevation.min())
        normalized /= float(normalized.max()) + 1e-8
        self.model.hfield_data[:] = normalized.ravel(order="C")
        self.data = mujoco.MjData(self.model)
        self.renderer: Any = None
        self._step_count = 0

    def _xml(self) -> str:
        rows, cols = self.elevation.shape
        return f"""<mujoco model="terrain_vehicle">
          <option timestep="0.02" gravity="0 0 -9.81"/>
          <asset><hfield name="terrain" nrow="{rows}" ncol="{cols}" size="8 8 2 0.1"/></asset>
          <worldbody>
            <geom type="hfield" hfield="terrain" friction="1 0.5 0.1"/>
            <body name="chassis" pos="0 0 0.8"><freejoint/><geom type="box" size=".55 .4 .15" mass="5"/>
              <body name="wheel_fl" pos=".4 .43 -.25"><joint name="wheel_fl" type="hinge" axis="0 1 0"/><geom type="cylinder" size=".18 .08" euler="90 0 0" mass=".3"/></body>
              <body name="wheel_fr" pos=".4 -.43 -.25"><joint name="wheel_fr" type="hinge" axis="0 1 0"/><geom type="cylinder" size=".18 .08" euler="90 0 0" mass=".3"/></body>
              <body name="wheel_rl" pos="-.4 .43 -.25"><joint name="wheel_rl" type="hinge" axis="0 1 0"/><geom type="cylinder" size=".18 .08" euler="90 0 0" mass=".3"/></body>
              <body name="wheel_rr" pos="-.4 -.43 -.25"><joint name="wheel_rr" type="hinge" axis="0 1 0"/><geom type="cylinder" size=".18 .08" euler="90 0 0" mass=".3"/></body>
            </body>
          </worldbody>
          <actuator><motor joint="wheel_fl" gear="2"/><motor joint="wheel_fr" gear="2"/>
            <motor joint="wheel_rl" gear="2"/><motor joint="wheel_rr" gear="2"/></actuator>
        </mujoco>"""

    def _obs(self) -> dict[str, np.ndarray]:
        center = np.asarray(self.elevation.shape) // 2
        radius = self.patch_size // 2
        padded = np.pad(self.elevation, radius, mode="edge")
        patch = padded[center[0]:center[0] + self.patch_size, center[1]:center[1] + self.patch_size]
        return {"patch": patch[None].astype(np.float32),
                "orientation": self.data.qpos[3:7].astype(np.float32),
                "velocity": self.data.qvel[:6].astype(np.float32)}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        _require_deps()
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0.0, 0.0, 1.0]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (4,):
            raise ValueError("action must contain four wheel torques")
        self.data.ctrl[:] = np.clip(values, -1.0, 1.0)
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        speed = float(np.linalg.norm(self.data.qvel[:2]))
        reward = speed - 0.01 * float(np.square(values).sum())
        terminated = bool(self.data.qpos[2] < 0.1)
        truncated = self._step_count >= self.max_steps
        return self._obs(), reward, terminated, truncated, {"speed": speed}

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


TerrainVehicleEnv = TerrainSkidSteerEnv
