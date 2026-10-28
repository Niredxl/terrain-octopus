"""Common interface and derived statistics for 2-D terrain heightmaps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class HeightmapTerrain(ABC):
    """Abstract terrain source returning standardized finite 2-D arrays.

    The slope is the gradient magnitude
    ``sqrt((dh/dx)**2 + (dh/dy)**2)`` computed with centred finite
    differences. Roughness is the local variance
    ``E[(h - E[h])**2]`` over a square neighbourhood.
    """

    def __init__(self, size: Tuple[int, int] = (64, 64), resolution: float = 1.0,
                 seed: Optional[int] = None) -> None:
        if len(size) != 2 or min(size) < 2:
            raise ValueError("size must contain two dimensions >= 2")
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.size = (int(size[0]), int(size[1]))
        self.resolution = float(resolution)
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def generate_elevation(self) -> np.ndarray:
        """Return a float32 elevation array with shape ``size``."""

    def elevation(self) -> np.ndarray:
        """Generate and validate the heightmap."""
        values = np.asarray(self.generate_elevation(), dtype=np.float32)
        if values.shape != self.size or not np.isfinite(values).all():
            raise ValueError(f"elevation must be finite with shape {self.size}")
        return values

    def slope(self, elevation: Optional[np.ndarray] = None) -> np.ndarray:
        """Return gradient magnitude ``sqrt((dh/dx)^2 + (dh/dy)^2)``."""
        heightmap = self.elevation() if elevation is None else np.asarray(elevation, dtype=np.float32)
        if heightmap.shape != self.size:
            raise ValueError(f"elevation must have shape {self.size}")
        dy, dx = np.gradient(heightmap, self.resolution, self.resolution)
        return np.hypot(dx, dy).astype(np.float32)

    def roughness(self, elevation: Optional[np.ndarray] = None, radius: int = 1) -> np.ndarray:
        """Return local variance ``E[(h - E[h])^2]`` in a square window."""
        if radius < 1:
            raise ValueError("radius must be at least one")
        heightmap = self.elevation() if elevation is None else np.asarray(elevation, dtype=np.float32)
        if heightmap.shape != self.size:
            raise ValueError(f"elevation must have shape {self.size}")
        width = 2 * radius + 1
        padded = np.pad(heightmap, radius, mode="reflect")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (width, width))
        return windows.var(axis=(-2, -1)).astype(np.float32)

    def generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return elevation, slope, and roughness arrays."""
        elevation = self.elevation()
        return elevation, self.slope(elevation), self.roughness(elevation)


class ArrayTerrain(HeightmapTerrain):
    """Small adapter for passing a precomputed heightmap to consumers."""

    def __init__(self, elevation: np.ndarray, resolution: float = 1.0,
                 name: str = "terrain") -> None:
        values = np.asarray(elevation, dtype=np.float32)
        super().__init__(values.shape, resolution)
        self._elevation = values
        self.name = name

    def generate_elevation(self) -> np.ndarray:
        return self._elevation.copy()
