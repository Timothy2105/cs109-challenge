const ScoreBoard = ({ userScore, aiScore }) => {
  return (
    <div className="score-board">
      <div className="score">
        <h3>You</h3>
        <div className="score-number">{userScore}</div>
      </div>
      <div className="score">
        <h3>Mini-Timothy</h3>
        <div className="score-number">{aiScore}</div>
      </div>
    </div>
  );
};

export default ScoreBoard;