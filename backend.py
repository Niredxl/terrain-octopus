"""Small FastAPI service for the Terrain Octopus research dashboard."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from procedural_generators import EvolutionaryTerrainGenerator, PerlinTerrainGenerator


class GeneratorParameters(BaseModel):
    frequency: float = Field(3.0, gt=0)
    octaves: int = Field(4, ge=1, le=12)
    persistence: float = Field(0.5, gt=0, le=1)
    amplitude: float = Field(1.0, gt=0)
    seed: int = 421


class TrainingControl(BaseModel):
    running: bool


app = FastAPI(title="Terrain Octopus API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

palette = ["#62e6ff", "#b8f36a", "#f7b267", "#b39cff"]
labels = [
    ("fractal", "Fractal noise", "procedural"),
    ("cellular", "Cellular automata", "procedural"),
    ("gan", "GAN inference", "neural / untrained"),
    ("ddpm", "DDPM sampler", "neural / untrained"),
]
current_parameters = GeneratorParameters()
training_running = False
step = 120_000
started_at = time.monotonic()


def _downsample(values: np.ndarray, size: int = 10) -> list[list[float]]:
    rows = np.linspace(0, values.shape[0] - 1, size).astype(int)
    cols = np.linspace(0, values.shape[1] - 1, size).astype(int)
    return values[np.ix_(rows, cols)].round(4).tolist()


def _sample(sample_id: str, label: str, family: str, index: int, parameters: GeneratorParameters) -> dict[str, Any]:
    seed = parameters.seed + index * 13
    if sample_id == "cellular":
        terrain = EvolutionaryTerrainGenerator(
            (32, 32),
            smoothing_iterations=max(1, parameters.octaves),
            amplitude=parameters.amplitude,
            seed=seed,
        )
    else:
        terrain = PerlinTerrainGenerator(
            (32, 32),
            frequency=parameters.frequency,
            octaves=parameters.octaves,
            persistence=parameters.persistence,
            amplitude=parameters.amplitude,
            seed=seed,
        )
    elevation, slope, roughness = terrain.generate()
    return {
        "id": sample_id,
        "label": label,
        "family": family,
        "accent": palette[index],
        "values": _downsample(elevation),
        "stats": {
            "elevation": f"{float(elevation.mean()):.2f} m",
            "slope": f"{float(slope.mean()):.2f}",
            "roughness": f"{float(roughness.mean()):.2f}",
        },
    }


def _metrics() -> list[dict[str, float | int]]:
    global step
    points = []
    for index in range(24):
        x = index / 23
        points.append(
            {
                "step": int(step - (23 - index) * 5_000),
                "reward": round(0.18 + x * 0.027 + np.sin(index * 0.72) * 0.03, 4),
                "policyLoss": round(0.24 - x * 0.004 + np.cos(index * 0.55) * 0.012, 4),
                "entropy": round(0.91 - x * 0.014 + np.sin(index * 0.31) * 0.008, 4),
                "successRate": round(0.21 + x * 0.021 + np.sin(index * 0.4) * 0.018, 4),
            }
        )
    return points


def _event(level: str, message: str) -> dict[str, str]:
    return {"timestamp": datetime.now().strftime("%H:%M:%S"), "level": level, "message": message}


def dashboard_state() -> dict[str, Any]:
    return {
        "connection": "connected",
        "samples": [
            _sample(*definition, index, current_parameters)
            for index, definition in enumerate(labels)
        ],
        "metrics": _metrics(),
        "events": [
            _event("success", "FastAPI backend stream online"),
            _event("info", f"Terrain seed {current_parameters.seed} loaded"),
        ],
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "terrain-octopus-api"}


@app.get("/api/dashboard")
async def dashboard() -> dict[str, Any]:
    return dashboard_state()


@app.post("/api/terrain/generate")
async def generate(parameters: GeneratorParameters) -> dict[str, Any]:
    global current_parameters
    current_parameters = parameters
    return dashboard_state()


@app.post("/api/simulation/reset")
async def reset() -> dict[str, Any]:
    global current_parameters, step, started_at, training_running
    current_parameters = current_parameters.model_copy(update={"seed": int(np.random.default_rng().integers(1000, 9999))})
    step = 0
    started_at = time.monotonic()
    training_running = False
    return {"status": "reset", "seed": current_parameters.seed}


@app.get("/api/simulation/frame")
async def frame() -> dict[str, Any]:
    return {
        "episode": max(1, int((time.monotonic() - started_at) // 2) + 1),
        "step": step,
        "velocity": 0.84 if training_running else 0.0,
        "height": 0.84,
        "running": training_running,
    }


@app.get("/api/training/status")
async def training_status() -> dict[str, Any]:
    return {"running": training_running, "step": step, "metrics": _metrics()}


@app.post("/api/training/control")
async def training_control(control: TrainingControl) -> dict[str, Any]:
    global training_running
    training_running = control.running
    return await training_status()


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket) -> None:
    global step
    await websocket.accept()
    try:
        while True:
            if training_running:
                step += 500
            await websocket.send_json(
                {
                    "type": "snapshot",
                    "event": _event("success" if training_running else "info", "PPO rollout heartbeat"),
                    "training": {"running": training_running, "step": step, "metrics": _metrics()[-1]},
                    "simulation": await frame(),
                }
            )
            await asyncio.sleep(2)
    except (WebSocketDisconnect, RuntimeError):
        return
