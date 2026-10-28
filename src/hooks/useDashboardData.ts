import { useCallback, useMemo, useState } from "react";
import {
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

  const regenerate = useCallback((next: GeneratorParameters = parameters) => {
    setParameters(next);
    setState(createDemoState(next));
  }, [parameters]);

  const pushEvent = useCallback((event: StreamEvent) => {
    setState((current) => ({ ...current, events: [event, ...current.events].slice(0, 8) }));
  }, []);

  const toggleTraining = useCallback(() => {
    setRunning((current) => {
      const next = !current;
      pushEvent({
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
        level: next ? "success" : "warn",
        message: next ? "PPO training loop resumed" : "PPO training loop paused",
      });
      return next;
    });
  }, [pushEvent]);

  const reset = useCallback(() => {
    setRunning(false);
    regenerate({ ...parameters, seed: Math.floor(Math.random() * 9000) + 1000 });
    pushEvent({
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
      level: "info",
      message: "Simulation reset · new terrain seed queued",
    });
  }, [parameters, pushEvent, regenerate]);

  return useMemo(
    () => ({ parameters, setParameters, state, running, regenerate, toggleTraining, reset }),
    [parameters, state, running, regenerate, toggleTraining, reset],
  );
}
