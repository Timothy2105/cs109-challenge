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
  showResult,
  hasSubmitted  // Add this prop
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
          hasSubmitted={hasSubmitted}  // Pass it to PredictionCard
        />
        
        <UserInput
          composers={composers}
          selectedComposer={selectedComposer}
          onComposerSelect={onComposerSelect}
          onSubmit={onSubmit}
          result={result}
          showResult={showResult}
          hasSubmitted={hasSubmitted}  // You might also want to pass it to UserInput to disable buttons
        />
      </div>
    </div>
  );
};

export default GameArea;