const AudioPlayer = ({ audioSrc }) => {
  const isYouTubeEmbed = audioSrc && audioSrc.includes('youtube.com/embed/');

  const getHiddenYouTubeUrl = (url) => {
    // Keep minimal parameters that still work
    const hideParams = 'modestbranding=1&rel=0&controls=1';
    return url.includes('?') ? `${url}&${hideParams}` : `${url}?${hideParams}`;
  };

  if (isYouTubeEmbed) {
    return (
      <div className="audio-player">
        <p>Listen to this musical excerpt:</p>
        <div style={{ 
          position: 'relative', 
          width: '100%', 
          height: '315px',
          overflow: 'hidden',
          backgroundColor: '#181818',
          borderRadius: '8px'
        }}>
          <iframe 
            width="100%" 
            height="315" 
            src={getHiddenYouTubeUrl(audioSrc)}
            title="Musical excerpt"
            frameBorder="0" 
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
            referrerPolicy="strict-origin-when-cross-origin"
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              border: 'none'
            }}
          />
          {/* Overlay that covers everything but allows clicks through */}
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: '#181818',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            zIndex: 1,
            borderRadius: '8px',
            pointerEvents: 'none'
          }}>
            {/* Play button icon */}
            <div style={{
              fontSize: '48px',
              color: '#fff',
              marginBottom: '16px'
            }}>
              ▶️
            </div>
            {/* Text content */}
            <div style={{
              color: '#fff',
              fontSize: '18px',
              fontWeight: '500',
              textAlign: 'center',
              marginBottom: '8px',
              fontFamily: 'system-ui, -apple-system, sans-serif'
            }}>
              🎵 Classical Music Excerpt
            </div>
            <div style={{
              color: '#aaa',
              fontSize: '14px',
              textAlign: 'center',
              marginBottom: '8px',
              fontFamily: 'system-ui, -apple-system, sans-serif'
            }}>
              Click to play and listen carefully to identify the composer
            </div>
            <div style={{
              color: '#ff6b6b',
              fontSize: '12px',
              textAlign: 'center',
              fontFamily: 'system-ui, -apple-system, sans-serif'
            }}>
              ⚠️ Only click once to avoid bugs
            </div>
          </div>
        </div>
      </div>
    );
  }

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