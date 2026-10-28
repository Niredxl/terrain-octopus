export function SimulationViewport({ running }: { running: boolean }) {
  return (
    <div className="simulation-viewport">
      <div className="sim-grid" />
      <div className="sim-radar" />
      <div className="sim-robot">
        <div className="robot-body"><span /><span /><span /><span /></div>
        <div className="robot-shadow" />
      </div>
      <div className="sim-label sim-label-a">AGENT / 04</div>
      <div className="sim-label sim-label-b">Z +0.84</div>
      <div className="sim-label sim-label-c">PATCH 09×09</div>
      <div className="sim-crosshair">+</div>
      <div className="sim-status"><span className={`status-dot ${running ? "live" : ""}`} />{running ? "LIVE ROLLOUT" : "STANDBY"} <b>00:14:32</b></div>
    </div>
  );
}
