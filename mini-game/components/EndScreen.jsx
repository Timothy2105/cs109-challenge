const EndScreen = ({ userScore, aiScore, onPlayAgain }) => {
  const getEndMessage = () => {
    if (userScore > aiScore) {
      return {
        title: '🎉 You Won! 🎉',
        message: `Incredible! You beat Mini-Timmy ${userScore}-${aiScore}!

🏆 You have a better ear for classical music than our AI! Mini-Timmy is crying into his circuits right now... 😭🤖`
      };
    } else if (userScore < aiScore) {
      return {
        title: '🤖 Mini-Timmy Wins! 🤖',
        message: `Mini-Timmy defeated you ${aiScore}-${userScore}!

😅 Don't worry, Mini-Timmy has been trained on thousands of pieces! He's probably cheating anyway... those algorithms are so sneaky! 🕵️‍♂️`
      };
    } else {
      return {
        title: '🤝 It\'s a Tie! 🤝',
        message: `You tied with Mini-Timmy ${userScore}-${aiScore}!

🤯 This is rare! You and Mini-Timmy are perfectly matched! He's requesting a rematch... 🥊🤖`
      };
    }
  };

  const { title, message } = getEndMessage();

  return (
    <div className="end-screen">
      <h2>{title}</h2>
      <div className="final-message">
        {message.split('\n\n').map((paragraph, index) => (
          <div key={index} style={{ marginBottom: '1rem' }}>
            {paragraph}
          </div>
        ))}
      </div>
      <button className="play-again-btn" onClick={onPlayAgain}>
        Play Again
      </button>
    </div>
  );
};

export default EndScreen;