const PredictionCard = ({ aiPrediction, features }) => {
  return (
    <div className="prediction-card">
      <h3>🤖 Mini-Timmy's Prediction</h3>
      <div className="composer-prediction">{aiPrediction}</div>
      
      <div className="features-grid">
        <div className="feature-category">
          <h4>Pitch Classes (C-B)</h4>
          <div className="feature-list">
            {features.pitchClasses}
          </div>
        </div>
        
        <div className="feature-category">
          <h4>Top Chords</h4>
          <div className="feature-list">
            {features.chords}
          </div>
        </div>
        
        <div className="feature-category">
          <h4>Rhythm Features</h4>
          <div 
            className="feature-list"
            dangerouslySetInnerHTML={{ __html: features.rhythm }}
          />
        </div>
      </div>
    </div>
  );
};

export default PredictionCard;