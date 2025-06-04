import AudioPlayer from './AudioPlayer';
import PredictionCard from './PredictionCard';
import UserInput from './UserInput';

const GameArea = ({ 
  currentRound, 
  currentData, 
  composers, 
  selectedComposer, 
  onComposerSelect, 
  onSubmit, 
  result,
  showResult 
}) => {
  return (
    <div className="game-area">
      <div className="round-info">
        <h2>Round {currentRound} of 5</h2>
      </div>

      <AudioPlayer audioSrc={currentData.audio} />

      <div className="prediction-section">
        <PredictionCard 
          aiPrediction={currentData.aiPrediction}
          features={currentData.features}
        />
        
        <UserInput
          composers={composers}
          selectedComposer={selectedComposer}
          onComposerSelect={onComposerSelect}
          onSubmit={onSubmit}
          result={result}
          showResult={showResult}
        />
      </div>
    </div>
  );
};

export default GameArea;