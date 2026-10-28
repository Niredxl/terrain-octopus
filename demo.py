"""Plot all four terrain generators and run a short random-action episode."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from neural_generators import DiffusionTerrainGenerator, GANTerrainGenerator
from procedural_generators import EvolutionaryTerrainGenerator, PerlinTerrainGenerator


def main() -> None:
    generators = [PerlinTerrainGenerator(seed=1), EvolutionaryTerrainGenerator(seed=2),
                  GANTerrainGenerator(seed=3), DiffusionTerrainGenerator(seed=4)]
    figure, axes = plt.subplots(2, 2, subplot_kw={"projection": "3d"}, figsize=(12, 10))
    y, x = np.meshgrid(np.arange(generators[0].size[1]), np.arange(generators[0].size[0]))
    for axis, generator in zip(axes.flat, generators):
        axis.plot_surface(y, x, generator.elevation(), cmap="terrain", linewidth=0)
        axis.set_title(type(generator).__name__)
    figure.tight_layout()
    plt.show(block=False)
    try:
        from sim_env import TerrainSkidSteerEnv
        environment = TerrainSkidSteerEnv(generators[0], render_mode="rgb_array")
        environment.reset(seed=0)
        for _ in range(100):
            _, _, terminated, truncated, _ = environment.step(environment.action_space.sample())
            environment.render()
            if terminated or truncated:
                break
        environment.close()
    except ImportError as exc:
        print(f"Skipping MuJoCo episode: {exc}")


if __name__ == "__main__":
    main()
