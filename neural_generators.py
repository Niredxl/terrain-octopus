"""Untrained PyTorch terrain generators for inference demonstrations."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from terrain_interface import HeightmapTerrain

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    torch = None
    Tensor = object
    nn = None
    _TORCH_ERROR = exc


def _require_torch() -> None:
    if torch is None:
        raise ImportError("neural generators require PyTorch") from _TORCH_ERROR


class GANTerrainGenerator(HeightmapTerrain):
    """Lightweight untrained GAN-style generator from a latent vector."""

    def __init__(self, size: Tuple[int, int] = (64, 64), latent_dim: int = 64,
                 seed: Optional[int] = None) -> None:
        _require_torch()
        super().__init__(size, seed=seed)
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.network = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, int(np.prod(self.size))),
            nn.Tanh(),
        ).eval()

    @torch.no_grad()
    def generate_elevation(self) -> np.ndarray:
        latent = torch.from_numpy(self.rng.standard_normal(self.latent_dim).astype(np.float32))
        return self.network(latent).reshape(self.size).cpu().numpy().astype(np.float32)


class _Denoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(16, 1, 3, padding=1))

    def forward(self, sample: Tensor, _timestep: Tensor) -> Tensor:
        return self.net(sample)


class DiffusionTerrainGenerator(HeightmapTerrain):
    """Minimal untrained DDPM sampler with a convolutional denoising network."""

    def __init__(self, size: Tuple[int, int] = (64, 64), diffusion_steps: int = 8,
                 seed: Optional[int] = None) -> None:
        _require_torch()
        super().__init__(size, seed=seed)
        if diffusion_steps < 1:
            raise ValueError("diffusion_steps must be positive")
        self.diffusion_steps = int(diffusion_steps)
        self.denoiser = _Denoiser().eval()

    @torch.no_grad()
    def generate_elevation(self) -> np.ndarray:
        sample = torch.from_numpy(self.rng.standard_normal((1, 1, *self.size)).astype(np.float32))
        for step in reversed(range(self.diffusion_steps)):
            timestep = torch.tensor([step], dtype=torch.long)
            sample = sample - self.denoiser(sample, timestep) / self.diffusion_steps
            if step:
                sample = sample * 0.97
        return sample[0, 0].cpu().numpy().astype(np.float32)


GANGenerator = GANTerrainGenerator
DDPMGenerator = DiffusionTerrainGenerator
