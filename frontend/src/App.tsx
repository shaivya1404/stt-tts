const App = () => {
  return (
    <div className="app-container">
      <h1 className="app-title">TTS/STT Platform Dashboard</h1>
      <p className="app-description">
        This Phase 1 scaffolding stitches together the backend API, ML microservices, and frontend for
        rapid iteration on the enterprise TTS/STT platform.
      </p>
      <div className="status-badge">
        <span className="status-indicator" />
        All core services are online
      </div>
    </div>
  );
};

export default App;
