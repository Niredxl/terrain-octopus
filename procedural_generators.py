"""Procedural terrain generators using fractal noise and cellular automata."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from terrain_interface import ArrayTerrain, HeightmapTerrain


class PerlinTerrainGenerator(HeightmapTerrain):
    """Generate continuous uneven terrain with multi-octave value noise."""

    def __init__(self, size: Tuple[int, int] = (64, 64), frequency: float = 3.0,
                 octaves: int = 4, persistence: float = 0.5,
                 lacunarity: float = 2.0, amplitude: float = 1.0,
                 seed: Optional[int] = None) -> None:
        super().__init__(size, seed=seed)
        if frequency <= 0 or octaves < 1 or not 0 < persistence <= 1 or lacunarity <= 0:
            raise ValueError("invalid noise parameters")
        self.frequency, self.octaves = frequency, int(octaves)
        self.persistence, self.lacunarity, self.amplitude = persistence, lacunarity, amplitude

    def _value_noise(self, frequency: float) -> np.ndarray:
        rows, cols = self.size
        coarse_shape = (max(2, int(rows / frequency) + 2), max(2, int(cols / frequency) + 2))
        coarse = self.rng.random(coarse_shape).astype(np.float32)
        # Bilinear interpolation avoids a SciPy dependency while retaining
        # smooth, deterministic low-frequency value noise.
        y = np.linspace(0, coarse_shape[0] - 1, rows)
        x = np.linspace(0, coarse_shape[1] - 1, cols)
        yi = np.floor(y).astype(int).clip(0, coarse_shape[0] - 2)
        xi = np.floor(x).astype(int).clip(0, coarse_shape[1] - 2)
        yf, xf = y - yi, x - xi
        top = coarse[yi[:, None], xi[None, :]] * (1 - xf)[None, :] + coarse[
            yi[:, None], xi[None, :] + 1
        ] * xf[None, :]
        bottom = coarse[yi[:, None] + 1, xi[None, :]] * (1 - xf)[None, :] + coarse[
            yi[:, None] + 1, xi[None, :] + 1
        ] * xf[None, :]
        return (top * (1 - yf)[:, None] + bottom * yf[:, None]).astype(np.float32)

    def generate_elevation(self) -> np.ndarray:
        result = np.zeros(self.size, dtype=np.float32)
        weight, amplitude, frequency = 0.0, 1.0, self.frequency
        for _ in range(self.octaves):
            result += amplitude * (self._value_noise(frequency) * 2.0 - 1.0)
            weight += amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity
        return (result / max(weight, 1e-8) * self.amplitude).astype(np.float32)


class EvolutionaryTerrainGenerator(HeightmapTerrain):
    """Generate rugged terrain by evolving a cellular-automata field."""

    def __init__(self, size: Tuple[int, int] = (64, 64), smoothing_iterations: int = 4,
                 initial_spawn_rate: float = 0.45, amplitude: float = 1.0,
                 seed: Optional[int] = None) -> None:
        super().__init__(size, seed=seed)
        if smoothing_iterations < 0 or not 0 <= initial_spawn_rate <= 1:
            raise ValueError("invalid cellular automata parameters")
        self.smoothing_iterations = int(smoothing_iterations)
        self.initial_spawn_rate, self.amplitude = initial_spawn_rate, amplitude

    def generate_elevation(self) -> np.ndarray:
        field = self.rng.random(self.size) < self.initial_spawn_rate
        for _ in range(self.smoothing_iterations):
            padded = np.pad(field.astype(np.int8), 1, mode="wrap")
            neighbours = sum(
                padded[i:i + self.size[0], j:j + self.size[1]]
                for i in range(3) for j in range(3) if (i, j) != (1, 1)
            )
            field = (neighbours >= 4) | (field & (neighbours >= 3))
        terrain = field.astype(np.float32)
        for _ in range(2):
            padded = np.pad(terrain, 1, mode="reflect")
            terrain = sum(
                padded[i:i + self.size[0], j:j + self.size[1]]
                for i in range(3) for j in range(3)
            ) / 9.0
        terrain = (terrain - terrain.mean()) / (terrain.std() + 1e-6)
        return (terrain * self.amplitude).astype(np.float32)


def fractal_noise(shape: Tuple[int, int] = (64, 64), octaves: int = 4,
                  persistence: float = 0.5, seed: Optional[int] = None) -> HeightmapTerrain:
    """Compatibility factory returning a Perlin terrain generator sample."""
    return ArrayTerrain(PerlinTerrainGenerator(shape, octaves=octaves,
                                                persistence=persistence, seed=seed).elevation(),
                        name="fractal-perlin")


def cellular_automata(shape: Tuple[int, int] = (64, 64), iterations: int = 4,
                      seed: Optional[int] = None) -> HeightmapTerrain:
    """Compatibility factory returning a cellular-automata terrain sample."""
    generator = EvolutionaryTerrainGenerator(shape, smoothing_iterations=iterations, seed=seed)
    return ArrayTerrain(generator.elevation(), name="cellular-automata")
