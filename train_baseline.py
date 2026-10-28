"""Train a PPO baseline on Perlin terrain with optional Weights & Biases logs."""

from __future__ import annotations

import os

from procedural_generators import PerlinTerrainGenerator
from sim_env import TerrainSkidSteerEnv


def train(total_timesteps: int = 100_000):
    """Train PPO for ``total_timesteps`` and save the resulting policy."""
    try:
        import wandb
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:
        raise ImportError("training requires stable-baselines3 and wandb") from exc

    class WandbCallback(BaseCallback):
        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                wandb.log({"episode/speed": info.get("speed", 0.0),
                           "train/step": self.num_timesteps})
            return True

    run = wandb.init(project=os.getenv("WANDB_PROJECT", "terrain-octopus"))
    env = TerrainSkidSteerEnv(PerlinTerrainGenerator())
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=WandbCallback())
    model.save("ppo_perlin_baseline")
    env.close()
    run.finish()
    return model


if __name__ == "__main__":
    train()
