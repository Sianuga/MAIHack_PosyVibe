import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Visualizer from './Components/Visualizer';

const WEBSOCKET_URL = 'ws://localhost:8000/ws';
const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [liveDataLog, setLiveDataLog] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('Connecting...');
  const [generationStatus, setGenerationStatus] = useState('idle'); // idle, generating, playing, paused
  const [isPlaying, setIsPlaying] = useState(false);
  const [analyser, setAnalyser] = useState(null);

  const socket = useRef(null);
  const audioRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceRef = useRef(null);

  const setupWebSocket = () => {
    console.log("Attempting to connect WebSocket...");
    socket.current = new WebSocket(WEBSOCKET_URL);

    socket.current.onopen = () => {
      console.log('WebSocket connection established.');
      setConnectionStatus('Connected');
    };

    socket.current.onmessage = (event) => {
      console.log('Received data from server:', event.data);
      const newData = JSON.parse(event.data);
      setLiveDataLog(prevData => [newData, ...prevData.slice(0, 50)]);

      // ** THIS IS THE KEY: Listen for the music URL **
      if (newData.processed_data?.music_url) {
        console.log("Music URL received:", newData.processed_data.music_url);
        loadAndPlayAudio(newData.processed_data.music_url);
      }
    };

    socket.current.onclose = () => {
      console.log('WebSocket connection closed.');
      setConnectionStatus('Disconnected. Will try to reconnect...');
      // Simple reconnect logic
      setTimeout(setupWebSocket, 3000);
    };

    socket.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('Connection Error');
    };
  }

  useEffect(() => {
    setupWebSocket();
    return () => {
      if (socket.current) {
        socket.current.close();
      }
    };
  }, []);

  const loadAndPlayAudio = (url) => {
    // Cleanup old audio resources to prevent memory leaks
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = '';
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect();
    }

    const audio = new Audio(url);
    audio.crossOrigin = 'anonymous';
    audio.loop = true;
    audioRef.current = audio;

    if (!audioContextRef.current) {
      const context = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = context;
    }
    const context = audioContextRef.current;
    
    if (context.state === 'suspended') {
      context.resume();
    }

    const source = context.createMediaElementSource(audio);
    sourceRef.current = source;

    const newAnalyser = context.createAnalyser();
    newAnalyser.fftSize = 1024;
    
    source.connect(newAnalyser);
    newAnalyser.connect(context.destination);

    setAnalyser(newAnalyser);
    
    audio.play()
      .then(() => {
        setGenerationStatus('playing');
        setIsPlaying(true);
      })
      .catch(e => {
        console.error("Audio playback failed:", e);
        setGenerationStatus('paused'); // Allow user to manually start
        setIsPlaying(false);
      });
  };
  
  const handleGenerateClick = async () => {
    if (generationStatus === 'generating') return;

    setGenerationStatus('generating');
    setIsPlaying(false);
    if (audioRef.current) {
      audioRef.current.pause();
    }
    
    try {
      await fetch(`${API_BASE_URL}/trigger-generation`, { method: 'POST' });
    } catch (error) {
      console.error("Failed to trigger generation:", error);
      setGenerationStatus('idle'); // Reset on error
      alert("Failed to connect to the backend. Please ensure all services are running.");
    }
  };
  
  const togglePlayPause = () => {
    if (!audioRef.current) return;

    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
      setGenerationStatus('paused');
    } else {
      audioContextRef.current.resume();
      audioRef.current.play();
      setIsPlaying(true);
      setGenerationStatus('playing');
    }
  };
  
  // Renders the log item for the side panel
  const renderProcessedData = (data) => {
    if (data && data.music_url) {
      return (
        <>
          <p style={{margin: '5px 0', color: '#00c4ff'}}>{data.message}</p>
          <p style={{margin: '5px 0'}}><strong>Track:</strong> <a href={data.music_url} target="_blank" rel="noopener noreferrer">{data.music_url.split('/').pop()}</a></p>
          <p style={{margin: '5px 0', fontStyle: 'italic', opacity: 0.7}}>Source Prompt: {data.original_prompt}</p>
        </>
      );
    }
    return <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>{JSON.stringify(data, null, 2)}</pre>;
  };

  return (
    <div style={{ display: 'flex', width: '100vw', height: '100vh', background: '#000005', color: 'white' }}>
      {/* Main Visualizer Area */}
      <div style={{ flex: 1, height: '100%', position: 'relative' }}>
        <AnimatePresence>
          {generationStatus === 'idle' && (
             <motion.div
              key="generate-button"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              transition={{ type: 'spring', stiffness: 200, damping: 20 }}
              style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', zIndex: 10 }}
            >
              <button onClick={handleGenerateClick} style={buttonStyles.generate}>
                Generate Music
              </button>
            </motion.div>
          )}

          {generationStatus === 'generating' && (
            <motion.div key="generating-state" style={overlayStyles} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <div style={spinnerStyles.loader}></div>
              <p style={{ marginTop: '20px', fontSize: '1.2em', letterSpacing: '1px' }}>AI is composing your track...</p>
            </motion.div>
          )}
        </AnimatePresence>
        
        {/* Render visualizer only when ready */}
        {(generationStatus === 'playing' || generationStatus === 'paused') && (
          <>
            <Visualizer analyser={analyser} isPlaying={isPlaying} />
            <div style={{ position: 'absolute', bottom: '30px', left: '50%', transform: 'translateX(-50%)', zIndex: 10 }}>
                <button onClick={togglePlayPause} style={buttonStyles.playPause}>
                  {isPlaying ? 'Pause' : 'Play'}
                </button>
            </div>
          </>
        )}
         {/* Logo Header */}
         <div style={{ position: 'absolute', top: '30px', left: '30px', zIndex: 10, display: 'flex', alignItems: 'center' }}>
            <img src="/PosyVibe_Final.png" alt="PosyVibe Logo" style={{ width: '50px', height: '50px', marginRight: '15px' }} />
            <div>
              <h1 style={{ margin: 0, fontSize: '24px' }}>PosyVibe</h1>
              <p style={{ margin: 0, color: 'rgba(255, 255, 255, 0.7)', fontSize: '14px' }}>Music therapy app</p>
            </div>
         </div>
      </div>

      {/* Side Panel for Logs */}
      <div style={{ width: '350px', height: '100%', background: '#111', padding: '20px', overflowY: 'auto', borderLeft: '1px solid #333', fontFamily: 'monospace' }}>
        <h2>Live Event Stream</h2>
        <p>Status: <span style={{ color: connectionStatus === 'Connected' ? '#4CAF50' : '#F44336' }}>{connectionStatus}</span></p>
        <hr style={{borderColor: '#444'}}/>
        {liveDataLog.length === 0 ? (
          <p>Waiting for events...</p>
        ) : (
          liveDataLog.map((item) => (
            <motion.div
              key={item.request_id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              style={{ backgroundColor: '#2a2a2a', padding: '15px', margin: '10px 0', borderRadius: '4px', borderLeft: '4px solid #00bcd4' }}
            >
              <strong>Request ID:</strong> {item.request_id} <br />
              {renderProcessedData(item.processed_data)}
            </motion.div>
          ))
        )}
      </div>
    </div>
  );
}

// Styles for UI elements
const overlayStyles = {
  position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
  display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center',
  background: 'rgba(0, 0, 5, 0.8)', backdropFilter: 'blur(5px)', zIndex: 10,
};

const buttonStyles = {
  generate: {
    padding: '20px 40px', fontSize: '1.5em', fontWeight: 'bold', color: 'white',
    background: 'linear-gradient(145deg, #eb00ff, #002bff)', border: 'none',
    borderRadius: '50px', cursor: 'pointer', transition: 'transform 0.2s',
    boxShadow: '0 10px 30px rgba(120, 0, 255, 0.4)',
  },
  playPause: {
    padding: '15px 35px', fontSize: '1.2em', color: 'white',
    background: 'rgba(255, 255, 255, 0.1)', backdropFilter: 'blur(10px)',
    border: '1px solid rgba(255, 255, 255, 0.2)', borderRadius: '30px', cursor: 'pointer',
  }
};

const spinnerStyles = {
  loader: {
    border: '5px solid rgba(255, 255, 255, 0.2)',
    borderTop: '5px solid #ffffff',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    animation: 'spin 1s linear infinite',
  },
  '@keyframes spin': {
    '0%': { transform: 'rotate(0deg)' },
    '100%': { transform: 'rotate(360deg)' },
  }
};
// Inject keyframes for spinner animation
const styleSheet = document.styleSheets[0];
const keyframes = `@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`;
if (styleSheet) {
    styleSheet.insertRule(keyframes, styleSheet.cssRules.length);
}

export default App;