const UserInput = ({ composers, selectedComposer, onComposerSelect, onSubmit, result, showResult, hasSubmitted }) => {
  return (
    <div className="prediction-card">
      <h3>🎯 Your Turn</h3>
      <p>Who do you think composed this piece?</p>

      <div className="user-input">
        <div className="composer-buttons">
          {composers.map((composer) => (
            <button
              key={composer}
              className={`composer-btn ${selectedComposer === composer ? 'selected' : ''}`}
              onClick={() => onComposerSelect(composer)}
              disabled={hasSubmitted}
            >
              {composer}
            </button>
          ))}
        </div>

        <button className="submit-btn" onClick={onSubmit} disabled={!selectedComposer || hasSubmitted}>
          Submit Prediction
        </button>

        <div className={`result ${result.type} ${showResult ? 'show' : ''}`}>{result.message}</div>
      </div>
    </div>
  );
};

export default UserInput;
