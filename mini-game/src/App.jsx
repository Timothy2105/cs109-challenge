import { useState, useEffect } from 'react';
import Header from './components/Header';
import ScoreBoard from './components/ScoreBoard';
import GameArea from './components/GameArea';
import EndScreen from './components/EndScreen';
import './App.css';

// Utility function to shuffle and pick N items
function getRandomSubset(arr, n) {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, n);
}

function App() {
  const [currentRound, setCurrentRound] = useState(1);
  const [userScore, setUserScore] = useState(0);
  const [aiScore, setAiScore] = useState(0);
  const [selectedComposer, setSelectedComposer] = useState(null);
  const [result, setResult] = useState({ message: '', type: '' });
  const [gameEnded, setGameEnded] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [hasSubmitted, setHasSubmitted] = useState(false);
  const [musicData, setMusicData] = useState(null);
  const [allMusicData, setAllMusicData] = useState(null);
  const [composers, setComposers] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('gameData.json')
      .then((res) => res.json())
      .then((data) => {
        setAllMusicData(data.musicData);
        const numRounds = 5;
        setMusicData(getRandomSubset(data.musicData, numRounds));
        setComposers(data.composers);
        setLoading(false);
      });
  }, []);

  if (loading || !musicData) {
    return (
      <div className="container">
        <h2>Loading game data...</h2>
      </div>
    );
  }

  const currentData = musicData[currentRound - 1];

  const handleComposerSelect = (composer) => {
    setSelectedComposer(composer);
    setResult({ message: '', type: '' });
    setShowResult(false);
  };

  const handleSubmit = () => {
    if (!selectedComposer) return;

    setHasSubmitted(true);

    const isUserCorrect = selectedComposer === currentData.answer;
    const isAiCorrect = currentData.aiPrediction === currentData.answer;

    // Update scores
    if (isUserCorrect) setUserScore((prev) => prev + 1);
    if (isAiCorrect) setAiScore((prev) => prev + 1);

    // Show result
    const resultMessage = isUserCorrect
      ? `✅ Correct! It was ${currentData.answer}`
      : `❌ Wrong! It was ${currentData.answer}`;

    setResult({
      message: resultMessage,
      type: isUserCorrect ? 'correct' : 'incorrect',
    });
    setShowResult(true);

    // Continue to next round or end game
    setTimeout(() => {
      if (currentRound < 5) {
        setCurrentRound((prev) => prev + 1);
        setSelectedComposer(null);
        setResult({ message: '', type: '' });
        setShowResult(false);
        setHasSubmitted(false);
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
    setHasSubmitted(false);
    // Pick 5 new random items from allMusicData
    if (allMusicData) {
      const numRounds = 5;
      setMusicData(getRandomSubset(allMusicData, numRounds));
    }
  };

  if (gameEnded) {
    return (
      <div className="container">
        <EndScreen userScore={userScore} aiScore={aiScore} onPlayAgain={resetGame} />
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
        hasSubmitted={hasSubmitted}
      />
    </div>
  );
}

export default App;
