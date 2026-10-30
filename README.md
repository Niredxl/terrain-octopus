# Terrain Octopus

Terrain Octopus is a small Python 3.10+ curriculum framework for generating
heightmap terrain and testing a simple four-wheel skid-steer agent on it. The
repository is intentionally lightweight: procedural terrain is NumPy-based,
while simulation, neural generators, and PPO training are optional layers.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`numpy` and `matplotlib` are enough for the procedural generators. Install
`torch` for the neural examples, and install `gymnasium` plus `mujoco` for the
vehicle environment. `stable-baselines3` and `wandb` are only needed for the
PPO baseline. The modules report an actionable `ImportError` when an optional
package is missing.

## Quick start

Generate a terrain and inspect its standardized arrays:

```python
from procedural_generators import fractal_noise

terrain = fractal_noise((64, 64), octaves=4, seed=7)
elevation, slope, roughness = terrain.generate()
print(elevation.shape, slope.mean(), roughness.mean())
```

Run the visualization demo:

```bash
python demo.py
```

The demo displays Perlin-style, cellular-automata, GAN, and diffusion samples
and then runs a short random-action MuJoCo episode when the simulation
dependencies are available.

## Modules

| Module | Purpose |
| --- | --- |
| `terrain_interface.py` | Abstract `HeightmapTerrain` interface, finite-array validation, slope, roughness, and `ArrayTerrain`. |
| `procedural_generators.py` | Multi-octave value/Perlin-style terrain and cellular-automata terrain without SciPy. |
| `neural_generators.py` | Small untrained PyTorch GAN-style and DDPM-style inference generators. |
| `sim_env.py` | Gymnasium environment with a MuJoCo hfield, four wheel torque actions, and patch/orientation/velocity observations. |
| `train_baseline.py` | Stable-Baselines3 PPO baseline with Weights & Biases logging. |
| `demo.py` | 2x2 terrain visualization and random-action episode. |

## Environment API

`TerrainSkidSteerEnv` exposes a continuous action vector with four wheel
torques in `[-1, 1]`. Observations are a dictionary containing:

- `patch`: local elevation patch shaped `(1, patch_size, patch_size)`;
- `orientation`: the vehicle quaternion;
- `velocity`: six-dimensional root velocity.

The environment follows Gymnasium's `reset()` and `step()` signatures and
supports `render_mode="human"` or `"rgb_array"`.

## PPO baseline

Run the default 100,000-timestep experiment:

```bash
python train_baseline.py
```

Set `WANDB_PROJECT` to change the W&B project name. The script saves the
trained policy as `ppo_perlin_baseline.zip`.

## Important notes

The GAN and diffusion models are deliberately **untrained demonstration
models**. They exercise the inference interfaces and produce synthetic arrays,
but their outputs are not expected to resemble realistic landscapes until
weights are trained or loaded.

The MuJoCo model is a compact educational skid-steer vehicle, not a calibrated
robot dynamics model. Terrain values are normalized into the MuJoCo hfield
after model creation, so the environment can consume generated heightmaps
without writing temporary XML or asset files.

## Validation

Compile all modules:

```bash
python -m py_compile terrain_interface.py procedural_generators.py \
  neural_generators.py sim_env.py train_baseline.py demo.py
```

The framework has also been smoke-tested for procedural elevation/slope/
roughness generation, neural inference, and MuJoCo environment reset/step
behavior.

## Live research dashboard

The repository includes a browser dashboard under `src/` for inspecting terrain
generators and PPO experiments. It is a Vite + React + TypeScript frontend with
a dark scientific-console UI, responsive terrain gallery, SVG simulation
viewport, live metric charts, generator controls, and a typed REST/WebSocket
client contract for a FastAPI backend.

Install Node dependencies and run the dashboard locally:

```bash
npm install
npm run dev
```

Start the live FastAPI service in a second terminal:

```bash
python -m pip install -r requirements.txt
python -m uvicorn backend:app --reload --port 8000
```

The dashboard connects to `http://127.0.0.1:8000` by default. Set
`VITE_API_URL` before `npm run dev` when the API is hosted elsewhere. The live
service exposes:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Health and service identity |
| `GET /api/dashboard` | Current terrain gallery, PPO metrics, and events |
| `POST /api/terrain/generate` | Generate terrain from frequency/octave/persistence/amplitude/seed |
| `POST /api/simulation/reset` | Reset the demo simulation and choose a new seed |
| `GET /api/simulation/frame` | Current episode, velocity, height, and run state |
| `GET /api/training/status` | Current PPO status and metrics |
| `POST /api/training/control` | Start or pause the training loop |
| `WS /ws/stream` | Two-second telemetry heartbeats with event, metric, and simulation snapshots |

If the API cannot be reached, the frontend clearly labels **DEMO FALLBACK** and
continues to provide interactive generated surfaces and controls.
