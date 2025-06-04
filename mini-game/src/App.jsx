import { useState } from 'react'
import Header from './components/Header'
import ScoreBoard from './components/ScoreBoard'
import GameArea from './components/GameArea'
import EndScreen from './components/EndScreen'
import './App.css'

const musicData = [
  {
    audio: '', // You'll replace this with actual audio file paths
    answer: 'Bach',
    aiPrediction: 'Bach',
    features: {
      pitchClasses: 'C: 45, C#: 12, D: 23, D#: 8, E: 34, F: 19, F#: 15, G: 28, G#: 7, A: 31, A#: 11, B: 18',
      chords: 'C:maj: 8, G:maj: 6, F:maj: 4, A:min: 5, D:min: 3, E:min: 2',
      rhythm: '16th: 0.25, 8th: 0.40, Qtr: 0.30<br>Mean dur: 0.45, Var dur: 0.12<br>Mean IOI: 0.52, Var IOI: 0.08'
    }
  },
  {
    audio: '',
    answer: 'Mozart',
    aiPrediction: 'Chopin',
    features: {
      pitchClasses: 'C: 38, C#: 15, D: 42, D#: 6, E: 29, F: 33, F#: 8, G: 25, G#: 12, A: 36, A#: 9, B: 22',
      chords: 'G:maj: 7, C:maj: 5, D:maj: 6, F:maj: 3, A:min: 4, B:min: 2',
      rhythm: '16th: 0.35, 8th: 0.30, Qtr: 0.25<br>Mean dur: 0.38, Var dur: 0.15<br>Mean IOI: 0.48, Var IOI: 0.11'
    }
  },
  {
    audio: '',
    answer: 'Beethoven',
    aiPrediction: 'Beethoven',
    features: {
      pitchClasses: 'C: 52, C#: 8, D: 35, D#: 14, E: 28, F: 25, F#: 18, G: 41, G#: 5, A: 27, A#: 16, B: 31',
      chords: 'C:maj: 9, F:maj: 7, G:maj: 8, Bb:maj: 4, D:min: 6, G:min: 3',
      rhythm: '16th: 0.20, 8th: 0.45, Qtr: 0.28<br>Mean dur: 0.52, Var dur: 0.18<br>Mean IOI: 0.55, Var IOI: 0.14'
    }
  },
  {
    audio: '',
    answer: 'Chopin',
    aiPrediction: 'Debussy',
    features: {
      pitchClasses: 'C: 29, C#: 22, D: 18, D#: 19, E: 33, F: 16, F#: 25, G: 21, G#: 18, A: 24, A#: 13, B: 27',
      chords: 'F#:maj: 5, Db:maj: 4, Ab:maj: 6, C:min: 7, F:min: 3, Bb:min: 5',
      rhythm: '16th: 0.42, 8th: 0.28, Qtr: 0.22<br>Mean dur: 0.31, Var dur: 0.22<br>Mean IOI: 0.43, Var IOI: 0.16'
    }
  },
  {
    audio: '',
    answer: 'Debussy',
    aiPrediction: 'Mozart',
    features: {
      pitchClasses: 'C: 21, C#: 28, D: 15, D#: 24, E: 19, F: 32, F#: 17, G: 26, G#: 23, A: 18, A#: 20, B: 14',
      chords: 'Db:maj: 6, Gb:maj: 4, Ab:maj: 5, F:min: 4, C:min: 3, Eb:min: 7',
      rhythm: '16th: 0.18, 8th: 0.32, Qtr: 0.35<br>Mean dur: 0.68, Var dur: 0.25<br>Mean IOI: 0.71, Var IOI: 0.19'
    }
  }
];

const composers = ['Bach', 'Mozart', 'Beethoven', 'Chopin', 'Debussy', 'Brahms'];

function App() {
  const [currentRound, setCurrentRound] = useState(1);
  const [userScore, setUserScore] = useState(0);
  const [aiScore, setAiScore] = useState(0);
  const [selectedComposer, setSelectedComposer] = useState(null);
  const [result, setResult] = useState({ message: '', type: '' });
  const [gameEnded, setGameEnded] = useState(false);
  const [showResult, setShowResult] = useState(false);

  const currentData = musicData[currentRound - 1];

  const handleComposerSelect = (composer) => {
    setSelectedComposer(composer);
    setResult({ message: '', type: '' });
    setShowResult(false);
  };

  const handleSubmit = () => {
    if (!selectedComposer) return;

    const isUserCorrect = selectedComposer === currentData.answer;
    const isAiCorrect = currentData.aiPrediction === currentData.answer;

    // Update scores
    if (isUserCorrect) setUserScore(prev => prev + 1);
    if (isAiCorrect) setAiScore(prev => prev + 1);

    // Show result
    const resultMessage = isUserCorrect 
      ? `✅ Correct! It was ${currentData.answer}`
      : `❌ Wrong! It was ${currentData.answer}`;
    
    setResult({
      message: resultMessage,
      type: isUserCorrect ? 'correct' : 'incorrect'
    });
    setShowResult(true);

    // Continue to next round or end game
    setTimeout(() => {
      if (currentRound < 5) {
        setCurrentRound(prev => prev + 1);
        setSelectedComposer(null);
        setResult({ message: '', type: '' });
        setShowResult(false);
      } else {
        setGameEnded(true);
      }
    }, 2000);
  };

  const resetGame = () => {
    setCurrentRound(1);
    setUserScore(0);
    setAiScore(0);
    setSelectedComposer(null);
    setResult({ message: '', type: '' });
    setGameEnded(false);
    setShowResult(false);
  };

  if (gameEnded) {
    return (
      <div className="container">
        <EndScreen 
          userScore={userScore} 
          aiScore={aiScore} 
          onPlayAgain={resetGame} 
        />
      </div>
    );
  }

  return (
    <div className="container">
      <Header />
      
      <ScoreBoard userScore={userScore} aiScore={aiScore} />
      
      <GameArea
        currentRound={currentRound}
        currentData={currentData}
        composers={composers}
        selectedComposer={selectedComposer}
        onComposerSelect={handleComposerSelect}
        onSubmit={handleSubmit}
        result={result}
        showResult={showResult}
      />
    </div>
  );
}

export default App;