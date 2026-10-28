export type ConnectionState = "demo" | "connecting" | "connected" | "error";

export interface GeneratorParameters {
  frequency: number;
  octaves: number;
  persistence: number;
  amplitude: number;
  seed: number;
}

export interface TerrainSample {
  id: string;
  label: string;
  family: string;
  accent: string;
  values: number[][];
  stats: { elevation: string; slope: string; roughness: string };
}

export interface StreamEvent {
  timestamp: string;
  level: "info" | "success" | "warn";
  message: string;
}

export interface MetricsSnapshot {
  step: number;
  reward: number;
  policyLoss: number;
  entropy: number;
  successRate: number;
}

export interface DashboardState {
  connection: ConnectionState;
  samples: TerrainSample[];
  metrics: MetricsSnapshot[];
  events: StreamEvent[];
}

const terrainPalette = ["#62e6ff", "#b8f36a", "#f7b267", "#b39cff"];

function seededValue(seed: number, x: number, y: number): number {
  const value = Math.sin(seed * 12.9898 + x * 78.233 + y * 37.719) * 43758.5453;
  return value - Math.floor(value);
}

function buildSurface(seed: number, mode: number): number[][] {
  return Array.from({ length: 10 }, (_, y) =>
    Array.from({ length: 10 }, (_, x) => {
      const wave = Math.sin((x + seed) * 0.62) * 0.14 + Math.cos((y - seed) * 0.48) * 0.12;
      const ridge = Math.sin((x + y + mode) * 0.34) * 0.09;
      return Math.max(0.06, Math.min(0.98, 0.18 + seededValue(seed + mode, x, y) * 0.52 + wave + ridge));
    }),
  );
}

export function createDemoState(parameters: GeneratorParameters): DashboardState {
  const definitions = [
    ["fractal", "Fractal noise", "procedural", 0],
    ["cellular", "Cellular automata", "procedural", 1],
    ["gan", "GAN inference", "neural / untrained", 2],
    ["ddpm", "DDPM sampler", "neural / untrained", 3],
  ] as const;
  const samples = definitions.map(([id, label, family, mode], index) => {
    const seed = parameters.seed + index * 13 + Math.round(parameters.frequency);
    const values = buildSurface(seed, mode);
    const mean = values.flat().reduce((sum, value) => sum + value, 0) / 100;
    return {
      id,
      label,
      family,
      accent: terrainPalette[index],
      values,
      stats: {
        elevation: `${(mean * parameters.amplitude).toFixed(2)} m`,
        slope: `${(0.11 + index * 0.018 + parameters.persistence * 0.04).toFixed(2)}`,
        roughness: `${(0.07 + index * 0.025).toFixed(2)}`,
      },
    };
  });
  const metrics = Array.from({ length: 24 }, (_, index) => ({
    step: (index + 1) * 5_000,
    reward: 0.18 + index * 0.027 + Math.sin(index * 0.72) * 0.03,
    policyLoss: 0.24 - index * 0.004 + Math.cos(index * 0.55) * 0.012,
    entropy: 0.91 - index * 0.014 + Math.sin(index * 0.31) * 0.008,
    successRate: 0.21 + index * 0.021 + Math.sin(index * 0.4) * 0.018,
  }));
  const events: StreamEvent[] = [
    { timestamp: "09:42:18", level: "success", message: "PPO rollout batch 184 committed" },
    { timestamp: "09:42:16", level: "info", message: "Terrain patch normalized to [0, 1]" },
    { timestamp: "09:42:12", level: "info", message: "Vehicle reset · seed 0421" },
    { timestamp: "09:42:08", level: "warn", message: "Slope variance above target envelope" },
    { timestamp: "09:42:03", level: "success", message: "WebSocket demo stream online" },
  ];
  return { connection: "demo", samples, metrics, events };
}

export function createApiClient(baseUrl = "") {
  async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${baseUrl}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
    if (!response.ok) throw new Error(`Backend returned ${response.status}`);
    return response.json() as Promise<T>;
  }

  return {
    health: () => request<{ status: string }>("/health"),
    generateTerrain: (parameters: GeneratorParameters) =>
      request<TerrainSample>("/api/terrain/generate", {
        method: "POST",
        body: JSON.stringify(parameters),
      }),
    resetSimulation: () => request<{ status: string }>("/api/simulation/reset", { method: "POST" }),
    createStream: (onEvent: (event: StreamEvent) => void) => {
      const socket = new WebSocket(`${baseUrl.replace(/^http/, "ws")}/ws/stream`);
      socket.onmessage = (message) => onEvent(JSON.parse(message.data) as StreamEvent);
      return socket;
    },
  };
}
