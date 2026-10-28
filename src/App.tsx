import { useMemo, useState } from "react";
import { Panel } from "./components/Panel";
import { TerrainSurface } from "./components/TerrainSurface";
import { MetricsGrid } from "./components/MetricChart";
import { SimulationViewport } from "./components/SimulationViewport";
import { useDashboardData } from "./hooks/useDashboardData";

const architecture = [
  ["01", "terrain_interface.py", "Heightmap contract + statistics"],
  ["02", "procedural_generators.py", "Fractal noise + cellular fields"],
  ["03", "neural_generators.py", "GAN / DDPM inference adapters"],
  ["04", "sim_env.py", "MuJoCo skid-steer environment"],
  ["05", "train_baseline.py", "Stable-Baselines3 PPO loop"],
];

function App() {
  const dashboard = useDashboardData();
  const [activeTab, setActiveTab] = useState("Overview");
  const [showParameters, setShowParameters] = useState(true);
  const { parameters, setParameters, state } = dashboard;
  const runLabel = dashboard.running ? "Pause training" : "Start training";
  const connectionText = state.connection === "demo" ? "DEMO DATA" : state.connection.toUpperCase();
  const latestMetric = state.metrics[state.metrics.length - 1];
  const updateParameter = (key: keyof typeof parameters, value: number) => {
    const next = { ...parameters, [key]: value };
    setParameters(next);
    dashboard.regenerate(next);
  };
  const navItems = useMemo(() => ["Overview", "Generators", "PPO Lab", "Run history"], []);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark">◈</div><div><strong>TERRAIN<br />OCTOPUS</strong><span>research console</span></div></div>
        <div className="side-label">WORKSPACE</div>
        <nav>{navItems.map((item, index) => <button key={item} className={activeTab === item ? "active" : ""} onClick={() => setActiveTab(item)}><span className="nav-index">0{index + 1}</span>{item}<span className="nav-arrow">↗</span></button>)}</nav>
        <div className="sidebar-bottom">
          <div className="connection-card"><span className="status-dot live" /><div><small>BACKEND LINK</small><b>{connectionText}</b></div><span className="signal">⌁</span></div>
          <div className="workspace-id"><small>WORKSPACE ID</small><b>terrain-octopus / main</b></div>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div><span className="eyebrow">LIVE EXPERIMENT / 28 OCT 2026</span><h1>{activeTab === "Overview" ? "Terrain intelligence dashboard" : activeTab}</h1></div>
          <div className="topbar-actions"><span className="latency"><span className="status-dot live" />12 ms</span><button className="icon-button">?</button><div className="avatar">NA</div></div>
        </header>

        <div className="content">
          <div className="hero-row">
            <Panel title="Project context" eyebrow="01 / ARCHITECTURE" className="context-panel">
              <div className="context-copy"><p>Generate, inspect, and train skid-steer agents on synthetic elevation fields. This console mirrors the Python curriculum framework and is ready to connect to a FastAPI runtime.</p><span className="tag">PYTHON 3.10+</span><span className="tag">MUJOCO</span><span className="tag">PPO</span></div>
              <div className="architecture-list">{architecture.map(([number, file, description]) => <div className="architecture-row" key={file}><span className="arch-number">{number}</span><div><b>{file}</b><span>{description}</span></div><span className="arch-link">↗</span></div>)}</div>
            </Panel>
            <Panel title="Run control" eyebrow="02 / EXECUTION" className="control-panel">
              <div className="control-status"><span className={`status-dot ${dashboard.running ? "live" : ""}`} /><div><b>{dashboard.running ? "Training in progress" : "Ready for rollout"}</b><span>Local demo stream · seed {parameters.seed}</span></div></div>
              <div className="control-buttons"><button className="button button-primary" onClick={dashboard.toggleTraining}>{dashboard.running ? "Ⅱ" : "▶"} {runLabel}</button><button className="button button-secondary" onClick={dashboard.reset}>↺ Reset</button></div>
              <button className="parameter-toggle" onClick={() => setShowParameters((current) => !current)}><span>Generator parameters</span><span>{showParameters ? "−" : "+"}</span></button>
              {showParameters && <div className="sliders">
                <label><span>Frequency <b>{parameters.frequency.toFixed(1)}</b></span><input type="range" min="1" max="8" step=".1" value={parameters.frequency} onChange={(event) => updateParameter("frequency", Number(event.target.value))} /></label>
                <label><span>Octaves <b>{parameters.octaves}</b></span><input type="range" min="1" max="8" step="1" value={parameters.octaves} onChange={(event) => updateParameter("octaves", Number(event.target.value))} /></label>
                <label><span>Persistence <b>{parameters.persistence.toFixed(2)}</b></span><input type="range" min=".1" max=".9" step=".01" value={parameters.persistence} onChange={(event) => updateParameter("persistence", Number(event.target.value))} /></label>
                <label><span>Amplitude <b>{parameters.amplitude.toFixed(2)}</b></span><input type="range" min=".2" max="2" step=".05" value={parameters.amplitude} onChange={(event) => updateParameter("amplitude", Number(event.target.value))} /></label>
              </div>}
            </Panel>
          </div>

          <Panel title="Generator gallery" eyebrow="03 / TERRAIN SYNTHESIS" action={<div className="panel-action">4 generators <span>↗</span></div>}>
            <div className="terrain-grid">{state.samples.map((sample) => <TerrainSurface key={sample.id} sample={sample} />)}</div>
          </Panel>

          <div className="lower-grid">
            <Panel title="Simulation stream" eyebrow="04 / MUJOCO VIEWPORT" action={<span className="panel-action"><span className="status-dot live" /> stream active</span>}>
              <SimulationViewport running={dashboard.running} />
              <div className="sim-footer"><span>episode <b>184 / 500</b></span><span>reward <b className="text-lime">{latestMetric.reward.toFixed(3)}</b></span><span>velocity <b>0.84 m/s</b></span></div>
            </Panel>
            <Panel title="Live event stream" eyebrow="05 / TELEMETRY">
              <div className="event-stream">{state.events.map((event, index) => <div className="event-row" key={`${event.timestamp}-${index}`}><span className={`event-marker ${event.level}`} /><time>{event.timestamp}</time><span>{event.message}</span></div>)}</div>
              <div className="stream-footer"><span className="status-dot live" /> streaming from mock websocket <span>⌘ K</span></div>
            </Panel>
          </div>

          <Panel title="PPO training metrics" eyebrow="06 / OPTIMIZATION" action={<div className="metric-summary"><span>steps <b>120k</b></span><span>success <b className="text-lime">{Math.round(latestMetric.successRate * 100)}%</b></span><span>policy <b>v0.4.2</b></span></div>}>
            <MetricsGrid metrics={state.metrics} />
          </Panel>
          <footer className="page-footer"><span>Terrain Octopus / scientific robotics lab</span><span>Mock mode is active until a FastAPI backend is available.</span></footer>
        </div>
      </section>
    </main>
  );
}

export default App;
