import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import Visualizer from './Components/Visualizer';

// MODIFIED: Point WebSocket to the main backend server
const WEBSOCKET_URL = 'ws://localhost:8000/ws';

function App() {
  const [liveData, setLiveData] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('Connecting...');

  useEffect(() => {
    const socket = new WebSocket(WEBSOCKET_URL);

    socket.onopen = () => {
      console.log('WebSocket connection established with main backend.');
      setConnectionStatus('Connected');
    };

    socket.onmessage = (event) => {
      console.log('Received data from server:', event.data);
      const newData = JSON.parse(event.data);
      setLiveData(prevData => [newData, ...prevData]);
    };

    socket.onclose = () => {
      console.log('WebSocket connection closed.');
      setConnectionStatus('Disconnected');
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('Connection Error');
    };

    return () => {
      socket.close();
    };
  }, []);

  return (
    <div style={{ fontFamily: 'monospace', backgroundColor: '#1a1a1a', color: '#e0e0e0', minHeight: '100vh', padding: '20px' }}>
      <h1>Live Data Stream from Main Backend</h1>
      <p>Connection Status: <span style={{ color: connectionStatus === 'Connected' ? '#4CAF50' : '#F44336' }}>{connectionStatus}</span></p>
      
      <div style={{ marginTop: '20px', maxHeight: '400px', overflowY: 'auto', border: '1px solid #444', padding: '10px', borderRadius: '5px' }}>
        {liveData.length === 0 ? (
          <p>Waiting for data...</p>
        ) : (
          liveData.map((item) => (
            <motion.div
              key={item.request_id}
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              style={{ backgroundColor: '#2a2a2a', padding: '15px', margin: '10px 0', borderRadius: '4px', borderLeft: '4px solid #00bcd4' }}
            >
              <strong>Request ID:</strong> {item.request_id} <br />
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                {JSON.stringify(item.processed_data, null, 2)}
              </pre>
            </motion.div>
          ))
        )}
      </div>

          <div style={{ width: '100vw', height: '100vh', background: '#000' }}>
      <Visualizer />
    </div>
    </div>
  );
}

export default App;