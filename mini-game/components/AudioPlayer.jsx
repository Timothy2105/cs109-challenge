const AudioPlayer = ({ audioSrc }) => {
  return (
    <div className="audio-player">
      <p>Listen to this musical excerpt:</p>
      <audio controls key={audioSrc}>
        <source src={audioSrc} type="audio/mpeg" />
        Your browser does not support the audio element.
      </audio>
    </div>
  );
};

export default AudioPlayer;