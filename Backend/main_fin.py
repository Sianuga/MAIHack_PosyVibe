"""
Production EEG System - Ready for Integration
============================================

Complete EEG system with Thread 1 (Collection) + Thread 2 (Filtering)
Ready for Thread 3 (Your Emotion Detection Model) integration

HOW TO USE THIS SYSTEM:
======================

1. BASIC INTEGRATION:
   ```python
   from eeg_system import start_eeg_system, send_start_command, send_stop_command, get_result_from_system
   
   # Start system once at application startup
   config = {...}  # Your device config
   start_eeg_system(config)
   # Start collecting when needed
   send_start_command()
   
   # Get results continuously
   while working:
       result = get_result_from_system(timeout=1.0)
       if result:
           # result['result']['eeg_dataframe'] contains filtered EEG data
           process_eeg_data(result)
   
   # Stop when done
   send_stop_command()
   ```

2. FRONTEND INTEGRATION (Flask/FastAPI):
   ```python
   @app.post("/start-eeg")
   def start_eeg():
       send_start_command()
       return {"status": "started"}
   
   @app.post("/stop-eeg") 
   def stop_eeg():
       send_stop_command()
       return {"status": "stopped"}
   
   @app.get("/eeg-data")
   def get_eeg_data():
       result = get_result_from_system(timeout=0.1)
       return result if result else {"status": "no_data"}
   ```

3. THREAD 3 INTEGRATION:
   - Add your emotion detection model in the process_thread2_data() method
   - Replace the simulated model_result with your actual model predictions
   - See "ADD YOUR THREAD 3 HERE" markers below

DATA FORMAT YOU RECEIVE:
========================
result = {
    'result': {
        'eeg_dataframe': [
            {'time': 0.0, 'T8': 0.123, 'P4': 0.456, ...},  # 1st second
            {'time': 1.0, 'T8': 0.234, 'P4': 0.567, ...},  # 2nd second  
            {'time': 2.0, 'T8': 0.345, 'P4': 0.678, ...}   # 3rd second
        ],
        'channel_names': ['T8', 'P4', 'C4', 'Pz', 'Fz', 'Oz', 'F4', 'F3', 'P3', 'F8', 'PO3', 'C3', 'T7', 'F7'],
        'emotion_prediction': 'your_model_output',
        'confidence': 0.85
    }
}
"""

import threading
import queue
import time
import signal
import sys
from typing import Optional, Dict, Any, List
import logging
from enum import Enum
import pandas as pd

# Import working Thread 1 and Thread 2
from thread1_eeg_connector import BrainAccessEEGCollector, EEGDataPackage
from thread2_eeg_processor import EEGFilterProcessor, ProcessedEEGData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemCommand(Enum):
    """Commands that can be sent to the system"""
    START = "START"
    STOP = "STOP"
    SHUTDOWN = "SHUTDOWN"
    STATUS = "STATUS"

class SystemState(Enum):
    """Current state of the system"""
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING" 
    READY = "READY"
    WORKING = "WORKING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"

class EEGSystemController:
    """
    Production EEG System Controller
    
    Manages Thread 1 (EEG Collection) + Thread 2 (Filtering) + Thread 3 (Your Model)
    """
    
    def __init__(self, device_name: str, config: dict):
        self.device_name = device_name
        self.config = config
        
        # Thread references
        self.eeg_collector = None      # Thread 1 (EEG Collection)
        self.filter_processor = None   # Thread 2 (Filtering & Processing)
        # ADD YOUR THREAD 3 HERE: self.emotion_model = None
        
        # Communication queues
        self.command_queue = queue.Queue(maxsize=10)       # External -> Main
        self.raw_eeg_queue = queue.Queue(maxsize=100)      # Thread 1 -> Thread 2
        self.filtered_eeg_queue = queue.Queue(maxsize=50)  # Thread 2 -> Thread 3
        self.external_output_queue = queue.Queue(maxsize=50) # Main -> External System
        
        # System state
        self.current_state = SystemState.IDLE
        self.state_lock = threading.Lock()
        self.system_initialized = threading.Event()
        self.shutdown_event = threading.Event()
        
        # Session tracking
        self.session_start_time = None
        self.session_data_count = 0
        self.total_batches_processed = 0
        
        # Setup signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and other shutdown signals"""
        logger.info("Shutdown signal received")
        self.send_command(SystemCommand.SHUTDOWN)
        
    def _set_state(self, new_state: SystemState):
        """Thread-safe state change"""
        with self.state_lock:
            old_state = self.current_state
            self.current_state = new_state
            logger.info(f"State change: {old_state.value} -> {new_state.value}")
    
    def get_state(self) -> SystemState:
        """Get current system state"""
        with self.state_lock:
            return self.current_state
    
    def send_command(self, command: SystemCommand, data: Any = None):
        """Send command to the system (thread-safe)"""
        try:
            command_data = {'command': command, 'data': data, 'timestamp': time.time()}
            self.command_queue.put_nowait(command_data)
        except queue.Full:
            logger.warning(f"Command queue full, dropping command: {command.value}")
    
    def get_external_output(self, timeout: float = None) -> Optional[Dict]:
        """
        Get results from the EEG system for your application
        
        Returns:
        --------
        Dict with filtered EEG data ready for emotion detection, or None if no data
        """
        try:
            if timeout is None:
                return self.external_output_queue.get_nowait()
            else:
                return self.external_output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def initialize_eeg_system(self) -> bool:
        """Initialize EEG hardware and processing threads"""
        self._set_state(SystemState.INITIALIZING)
        
        try:
            logger.info("Initializing EEG system...")
            
            # Initialize Thread 1 (EEG Collection)
            self.eeg_collector = BrainAccessEEGCollector(
                device_name=self.device_name,
                output_queue=self.raw_eeg_queue,
                cap=self.config['channels'],
                gain=self.config['gain'],
                sync_delay=self.config.get('sync_delay', 10.0)
            )
            
            # Initialize Thread 2 (Filtering & Processing)
            filter_config = self.config.get('filter_config', {
                'l_freq': 1.0,      # High-pass frequency
                'h_freq': 40,       # Low-pass frequency
                'notch_freq': 60    # Notch filter frequency
            })
            
            self.filter_processor = EEGFilterProcessor(
                input_queue=self.raw_eeg_queue,
                output_queue=self.filtered_eeg_queue,
                filter_config=filter_config
            )
            
            # ADD YOUR THREAD 3 INITIALIZATION HERE:
            # self.emotion_model = YourEmotionModel(
            #     input_queue=self.filtered_eeg_queue,
            #     output_queue=self.model_results_queue,
            #     model_config=self.config.get('model_config', {})
            # )
            
            # Start threads
            self.eeg_collector.start()
            self.filter_processor.start()
            # ADD YOUR THREAD 3 START HERE: self.emotion_model.start()
            
            # Wait for initialization
            if not self.eeg_collector.wait_for_connection(timeout=25):
                raise RuntimeError("Failed to connect to EEG device")
            
            if not self.eeg_collector.wait_for_streaming(timeout=15):
                raise RuntimeError("Failed to start EEG streaming")
            
            if not self.eeg_collector.wait_for_synchronization(timeout=15):
                raise RuntimeError("Failed to synchronize EEG device")
            
            # Check system status
            device_info = self.eeg_collector.get_device_info()
            logger.info(f"EEG ready: {device_info['valid_channels']}/{device_info['num_channels']} channels, "
                       f"{device_info['sample_rate']} Hz, {device_info['battery_level']}% battery")
            
            if not self.filter_processor.is_active():
                raise RuntimeError("Filter processor failed to start")
            
            logger.info("Filter processor ready")
            
            # ADD YOUR THREAD 3 STATUS CHECK HERE:
            # if not self.emotion_model.is_active():
            #     raise RuntimeError("Emotion model failed to start")
            # logger.info("Emotion model ready")
            
            self.system_initialized.set()
            self._set_state(SystemState.READY)
            logger.info("EEG system ready for commands")
            
            return True
            
        except Exception as e:
            logger.error(f"EEG system initialization failed: {e}")
            self._set_state(SystemState.ERROR)
            return False
    
    def start_working(self):
        """Start the working session (call this when user clicks START)"""
        if self.get_state() != SystemState.READY:
            logger.warning("System not ready, cannot start working")
            return False
        
        self._set_state(SystemState.WORKING)
        self.session_start_time = time.time()
        self.session_data_count = 0
        
        logger.info("EEG working session started")
        return True
    
    def stop_working(self):
        """Stop the working session (call this when user clicks STOP)"""
        if self.get_state() != SystemState.WORKING:
            logger.warning("System not working, cannot stop")
            return
        
        self._set_state(SystemState.STOPPING)
        
        # Log session statistics
        if self.session_start_time:
            session_duration = time.time() - self.session_start_time
            logger.info(f"Session ended: {session_duration:.1f}s, "
                       f"{self.session_data_count} results sent")
        
        self._set_state(SystemState.READY)
        logger.info("Working session stopped")
    
    def process_thread2_data(self):
        """
        Process filtered data from Thread 2
        
        THIS IS WHERE YOU ADD YOUR EMOTION DETECTION MODEL (Thread 3)
        """
        try:
            # Get processed DataFrame from Thread 2 (non-blocking)
            processed_data = self.filtered_eeg_queue.get_nowait()
            self.total_batches_processed += 1
            
            # Only process if we're in WORKING state
            if self.get_state() == SystemState.WORKING:
                
                df = processed_data.data_frame
                
                # ====================================================================
                # ADD YOUR THREAD 3 (EMOTION DETECTION MODEL) HERE:
                # ====================================================================
                
                # Option 1: Direct model processing (simple)
                # emotion_prediction, confidence = your_emotion_model.predict(df)
                
                # Option 2: Send to Thread 3 (advanced)
                # self.emotion_model.process_dataframe(processed_data)
                # emotion_result = self.emotion_results_queue.get(timeout=0.5)
                
                # For now, placeholder prediction:
                emotion_prediction = "neutral"  # Replace with: your_model.predict(df)
                confidence = 0.85               # Replace with: actual confidence score
                
                # ====================================================================
                
                # Create result package for your application
                model_result = {
                    'batch_id': processed_data.batch_id,
                    'timestamp': processed_data.timestamp,
                    'session_time_start': processed_data.session_time_start,
                    'session_time_end': processed_data.session_time_end,
                    'channels': len(processed_data.channel_names),
                    'channel_names': processed_data.channel_names,
                    
                    # Filtered EEG data ready for your model
                    'eeg_dataframe': df.to_dict('records'),  # List of dicts, one per second
                    'time_points': df['time'].tolist(),      # [0.0, 1.0, 2.0] (session relative)
                    
                    # Your model predictions
                    'emotion_prediction': emotion_prediction,
                    'confidence': confidence,
                    
                    # Optional: raw data access
                    'raw_eeg_values': {
                        col: df[col].tolist() for col in df.columns if col != 'time'
                    }
                }
                
                # Send result to your application
                external_output = {
                    'timestamp': time.time(),
                    'session_time': time.time() - self.session_start_time,
                    'result': model_result,
                    'session_count': self.session_data_count,
                    'system_state': self.get_state().value
                }
                
                try:
                    self.external_output_queue.put_nowait(external_output)
                    self.session_data_count += 1
                except queue.Full:
                    logger.warning("Output queue full, dropping result")
                
        except queue.Empty:
            pass  # No processed data available
    
    def run_command_listener(self):
        """Main command listening loop - runs in background thread"""
        logger.info("Starting EEG system...")
        
        # Initialize the system
        if not self.initialize_eeg_system():
            logger.error("Failed to initialize EEG system")
            return
        
        # Main processing loop
        while not self.shutdown_event.is_set():
            try:
                # Process commands
                try:
                    command_data = self.command_queue.get(timeout=0.1)
                    command = command_data['command']
                    
                    if command == SystemCommand.START:
                        if self.get_state() == SystemState.READY:
                            self.start_working()
                    
                    elif command == SystemCommand.STOP:
                        if self.get_state() == SystemState.WORKING:
                            self.stop_working()
                    
                    elif command == SystemCommand.SHUTDOWN:
                        logger.info("Shutdown requested")
                        break
                
                except queue.Empty:
                    pass
                
                # Process EEG data
                self.process_thread2_data()
                
                # Health checks
                if self.eeg_collector and not self.eeg_collector.is_connected():
                    logger.error("EEG device disconnected!")
                    self._set_state(SystemState.ERROR)
                    break
                
            except Exception as e:
                logger.error(f"Error in system loop: {e}")
                break
        
        # Cleanup
        self.shutdown()
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        status = {
            'state': self.get_state().value,
            'is_working': self.get_state() == SystemState.WORKING,
            'session_data_count': self.session_data_count,
            'session_duration': time.time() - self.session_start_time if self.session_start_time else 0
        }
        
        if self.eeg_collector:
            device_info = self.eeg_collector.get_device_info()
            status.update({
                'battery_level': device_info.get('battery_level'),
                'sample_rate': device_info.get('sample_rate'),
                'channels': device_info.get('valid_channels')
            })
        
        return status
    
    def shutdown(self):
        """Gracefully shutdown all threads"""
        logger.info("Shutting down EEG system...")
        
        self.shutdown_event.set()
        
        if self.get_state() == SystemState.WORKING:
            self.stop_working()
        
        # Stop threads in reverse order
        # ADD YOUR THREAD 3 SHUTDOWN HERE: 
        # if self.emotion_model: self.emotion_model.stop()
        
        if self.filter_processor:
            self.filter_processor.stop()
            self.filter_processor.join(timeout=5)
        
        if self.eeg_collector:
            self.eeg_collector.stop()
            self.eeg_collector.join(timeout=10)
        
        self._set_state(SystemState.IDLE)
        logger.info("EEG system shutdown complete")


# =============================================================================
# PUBLIC API - USE THESE FUNCTIONS IN YOUR APPLICATION
# =============================================================================

# Global system instance
_eeg_system: Optional[EEGSystemController] = None

def start_eeg_system(config: Dict) -> bool:
    """
    Start the EEG system (call once at application startup)
    
    Parameters:
    -----------
    config : Dict
        Configuration dictionary with:
        - 'device_name': BrainAccess device name (e.g., "BA MIDI 026")
        - 'channels': Channel mapping dict {electrode_id: name}
        - 'gain': Amplifier gain (4, 6, 8, or 12)
        - 'sync_delay': Synchronization delay in seconds (default: 10.0)
        - 'filter_config': Optional filter settings
    
    Returns:
    --------
    bool: True if successful, False if failed
    
    Example:
    --------
    config = {
        'device_name': "BA MIDI 026",
        'channels': {0: "T8", 2: "P4", 3: "C4", ...},
        'gain': 8,
        'sync_delay': 10.0
    }
    if start_eeg_system(config):
        print("EEG system ready!")
    """
    global _eeg_system
    
    if _eeg_system is not None:
        logger.warning("EEG system already running")
        return True
    
    _eeg_system = EEGSystemController(
        device_name=config['device_name'],
        config=config
    )
    
    # Start system in background thread
    system_thread = threading.Thread(
        target=_eeg_system.run_command_listener,
        name="EEGSystemController",
        daemon=True
    )
    system_thread.start()
    
    # Wait for initialization
    if _eeg_system.system_initialized.wait(timeout=45):
        logger.info("EEG system ready")
        return True
    else:
        logger.error("EEG system failed to initialize")
        return False

def send_start_command():
    """
    Start EEG data collection and processing
    
    Call this when user clicks "Start" button or when you want to begin
    collecting emotion detection data.
    """
    if _eeg_system:
        _eeg_system.send_command(SystemCommand.START)

def send_stop_command():
    """
    Stop EEG data collection and processing
    
    Call this when user clicks "Stop" button or when you want to stop
    collecting emotion detection data.
    """
    if _eeg_system:
        _eeg_system.send_command(SystemCommand.STOP)

def get_result_from_system(timeout: float = None) -> Optional[Dict]:
    """
    Get the latest emotion detection result from the EEG system
    
    Parameters:
    -----------
    timeout : float, optional
        Maximum time to wait for a result (seconds)
        - None: Return immediately (non-blocking)
        - 0.1: Wait up to 0.1 seconds
        - 1.0: Wait up to 1 second
    
    Returns:
    --------
    Dict or None:
        If data available, returns:
        {
            'result': {
                'eeg_dataframe': [
                    {'time': 0.0, 'T8': 0.123, 'P4': 0.456, ...},  # 1st second
                    {'time': 1.0, 'T8': 0.234, 'P4': 0.567, ...},  # 2nd second
                    {'time': 2.0, 'T8': 0.345, 'P4': 0.678, ...}   # 3rd second
                ],
                'channel_names': ['T8', 'P4', 'C4', ...],
                'emotion_prediction': 'happy',
                'confidence': 0.85,
                'time_points': [0.0, 1.0, 2.0]
            },
            'timestamp': 1234567890.123,
            'session_time': 15.5
        }
    
    Example:
    --------
    result = get_result_from_system(timeout=1.0)
    if result:
        eeg_data = result['result']['eeg_dataframe']
        emotion = result['result']['emotion_prediction']
        confidence = result['result']['confidence']
        print(f"Detected emotion: {emotion} ({confidence:.2f})")
    """
    if _eeg_system:
        return _eeg_system.get_external_output(timeout=timeout)
    return None

def get_system_status() -> Optional[Dict]:
    """
    Get current system status
    
    Returns:
    --------
    Dict with system information:
    {
        'state': 'WORKING',
        'is_working': True,
        'session_data_count': 42,
        'session_duration': 125.7,
        'battery_level': 65,
        'sample_rate': 250,
        'channels': 14
    }
    """
    if _eeg_system:
        return _eeg_system.get_system_status()
    return None

def shutdown_eeg_system():
    """
    Shutdown the EEG system (call when application exits)
    """
    global _eeg_system
    if _eeg_system:
        _eeg_system.send_command(SystemCommand.SHUTDOWN)
        time.sleep(2)
        _eeg_system = None


# =============================================================================
# EXAMPLE CONFIGURATIONS
# =============================================================================

# Standard 14-channel BrainAccess configuration
STANDARD_EEG_CONFIG = {
    'device_name': "BA MIDI 026",  # Replace with your device name
    'channels': {
        0: "T8", 2: "P4", 3: "C4", 4: "Pz", 5: "Fz", 6: "Oz", 7: "F4",
        8: "F3", 9: "P3", 10: "F8", 12: "PO3", 13: "C3", 14: "T7", 15: "F7"
    },
    'gain': 8,
    'sync_delay': 10.0,
    'filter_config': {
        'l_freq': 1.0,      # High-pass frequency
        'h_freq': 40,       # Low-pass frequency
        'notch_freq': 60    # Notch filter frequency
    }
}

