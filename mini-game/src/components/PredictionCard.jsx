const PredictionCard = ({ aiPrediction, features, hasSubmitted }) => {
  return (
    <div className={`prediction-card ai-prediction ${hasSubmitted ? 'show' : 'hidden'}`}>
      <h3>🤖 Mini-Timothy's Prediction</h3>
      <div className="composer-prediction">{hasSubmitted ? aiPrediction : 'Thinking...'}</div>

      <div className="features-grid">
        <div className="feature-category">
          <h4>Pitch Classes (C-B)</h4>
          <div className="feature-list">{features.pitchClasses}</div>
        </div>

        <div className="feature-category">
          <h4>Top Chords</h4>
          <div className="feature-list">{features.chords}</div>
        </div>
      </div>
    </div>
  );
};

export default PredictionCard;
