import { useCallback, useEffect, useMemo, useState } from "react";
import {
  createApiClient,
  createDemoState,
  type DashboardState,
  type GeneratorParameters,
  type StreamEvent,
} from "../api/client";

const defaultParameters: GeneratorParameters = {
  frequency: 3,
  octaves: 4,
  persistence: 0.5,
  amplitude: 1,
  seed: 421,
};

export function useDashboardData() {
  const [parameters, setParameters] = useState(defaultParameters);
  const [state, setState] = useState<DashboardState>(() => createDemoState(defaultParameters));
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const api = useMemo(() => createApiClient(import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000"), []);
  const pushEvent = useCallback((event: StreamEvent) => {
    setState((current) => ({ ...current, events: [event, ...current.events].slice(0, 8) }));
  }, []);

  useEffect(() => {
    let active = true;
    let socket: WebSocket | undefined;
    (async () => {
      try {
        const remote = await api.dashboard();
        if (!active) return;
        setState({ ...remote, connection: "connected" });
        setError(null);
        socket = api.createStream((event) => {
          if (active) pushEvent(event);
        });
        socket.onerror = () => {
          if (active) setError("Live stream disconnected; REST data remains available.");
        };
      } catch {
        if (active) {
          setState((current) => ({ ...current, connection: "demo" }));
          setError("Backend unavailable; showing demo data.");
        }
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
      socket?.close();
    };
  }, [api, pushEvent]);

  const regenerate = useCallback((next: GeneratorParameters = parameters) => {
    setParameters(next);
    void api.generateTerrain(next)
      .then((remote) => {
        setState({ ...remote, connection: "connected" });
        setError(null);
      })
      .catch(() => {
        setState(createDemoState(next));
        setError("Backend unavailable; showing demo data.");
      });
  }, [api, parameters]);

  const toggleTraining = useCallback(async () => {
    const next = !running;
    try {
      const result = await api.setTraining(next);
      setRunning(result.running);
      setState((current) => ({ ...current, connection: "connected", metrics: result.metrics }));
      setError(null);
    } catch {
      setRunning(next);
      pushEvent({
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
        level: next ? "success" : "warn",
        message: next ? "Demo PPO training loop resumed" : "Demo PPO training loop paused",
      });
    }
  }, [api, pushEvent, running]);

  const reset = useCallback(() => {
    void api.resetSimulation()
      .then(() => api.dashboard())
      .then((remote) => {
        setRunning(false);
        setState({ ...remote, connection: "connected" });
      })
      .catch(() => {
        setRunning(false);
        regenerate({ ...parameters, seed: Math.floor(Math.random() * 9000) + 1000 });
        pushEvent({
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
          level: "info",
          message: "Demo simulation reset · new terrain seed queued",
        });
      });
  }, [api, parameters, pushEvent, regenerate]);

  return useMemo(
    () => ({ parameters, setParameters, state, running, loading, error, regenerate, toggleTraining, reset }),
    [parameters, state, running, loading, error, regenerate, toggleTraining, reset],
  );
}
