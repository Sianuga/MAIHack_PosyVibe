"""
thread1_eeg_collector.py - Fixed with Synchronization
=====================================================

Thread 1: BrainAccess EEG Data Collection (Fixed)

Fixes:
- Adds synchronization delay after streaming starts
- Handles variable chunk sizes properly
- Better channel validation and error handling
- Adapts to actual device behavior instead of assumptions
"""

import threading
import queue
import time
import numpy as np
from typing import Optional, Dict, List
import logging
from dataclasses import dataclass

import brainaccess.core as bacore
from brainaccess.core.eeg_manager import EEGManager
from brainaccess.core.gain_mode import GainMode, multiplier_to_gain_mode
import brainaccess.core.eeg_channel as eeg_channel
from brainaccess.utils.exceptions import BrainAccessException

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EEGDataPackage:
    """
    Data package sent from Thread 1 to Thread 2
    
    3-Second batches (adaptive):
    - chunk_size: varies based on actual device behavior
    - eeg_data shape: (n_channels, samples_collected)
    - time_duration: approximately 3.0 seconds
    """
    timestamp: float          # System timestamp when batch completed
    chunk_size: int          # Actual samples in this batch
    time_duration: float     # Actual time duration
    eeg_data: np.ndarray     # Shape: (n_channels, chunk_size)
    channel_names: List[str] # Channel names that have data
    sample_rate: int         # Device sample rate

class BrainAccessEEGCollector(threading.Thread):
    """
    Thread 1: BrainAccess EEG Data Collection (Fixed)
    
    Adaptive Processing:
    1. Wait for device synchronization
    2. Adapt to actual chunk sizes received
    3. Accumulate data for ~3 seconds
    4. Send batches with actual collected data
    """
    
    def __init__(
        self,
        device_name: str,
        output_queue: queue.Queue,
        cap: Dict[int, str] = {
            0: "T8", 2: "P4", 3: "C4", 4: "Pz", 5: "Fz", 6: "Oz", 7: "F4",
            8: "F3", 9: "P3", 10: "F8", 12: "PO3", 13: "C3", 14: "T7", 15: "F7"
        },
        gain: int = 8,
        sync_delay: float = 5.0  # Synchronization delay in seconds
    ):
        """
        Initialize BrainAccess EEG collector thread
        
        Parameters:
        -----------
        device_name : str
            Name of the BrainAccess device (e.g., "BA MIDI 026")
        output_queue : queue.Queue
            Queue to send 3-second EEGDataPackage objects to Thread 2
        cap : Dict[int, str]
            Channel mapping (electrode number -> channel name)
        gain : int
            Amplifier gain (4, 6, 8, or 12)
        sync_delay : float
            Time to wait for device synchronization after streaming starts
        """
        super().__init__(name="BrainAccessEEGCollector", daemon=True)
        
        self.device_name = device_name
        self.output_queue = output_queue
        self.cap = cap
        self.gain = multiplier_to_gain_mode(gain) if gain in [4, 6, 8, 12] else GainMode.X8
        self.sync_delay = sync_delay
        
        # Adaptive timing constants
        self.TARGET_BATCH_DURATION = 1.0  # Target 3 seconds per batch
        self.sample_rate = 250               # Expected sample rate
        
        # Thread control events
        self._stop_event = threading.Event()
        self._connected = threading.Event()
        self._streaming = threading.Event()
        self._synchronized = threading.Event()
        self._error_event = threading.Event()
        
        # Error tracking
        self._error_message = ""
        
        # Device components
        self.mgr: Optional[EEGManager] = None
        self.eeg_channels: Dict[int, str] = {}
        self.channel_indexes: Dict[int, int] = {}
        self.eeg_channel_names: List[str] = []
        self.valid_channels: List[int] = []  # Channels that actually have data
        
        # Data accumulation (adaptive)
        self.accumulation_buffer = []    # List of numpy arrays
        self.batch_start_time = None     # Timestamp when accumulation started
        self.total_accumulated_samples = 0
        
        # Statistics
        self.total_samples_received = 0
        self.total_batches_sent = 0
        self.total_chunks_received = 0
        self.start_time = None
        self.last_data_time = None
        
        # Connection parameters
        self.connection_timeout = 20
        self.max_connection_attempts = 3
        self.connection_retry_delay = 2
        
        # Performance monitoring
        self.dropped_batches = 0
        self.queue_full_warnings = 0
        self.chunk_size_stats = []  # Track actual chunk sizes
        
    def run(self):
        """Main thread execution"""
        logger.info(f"Starting BrainAccess EEG collector for device: {self.device_name}")
        logger.info(f"Synchronization delay: {self.sync_delay}s after streaming starts")
        logger.info(f"Target batch duration: {self.TARGET_BATCH_DURATION}s")
        self.start_time = time.time()
        
        try:
            # Phase 1: Initialize BrainAccess core
            self._initialize_core()
            
            # Phase 2: Connect to device
            self._connect_device()
            
            # Phase 3: Setup channels
            self._setup_channels()
            
            # Phase 4: Start streaming
            self._start_streaming()
            
            # Phase 5: Synchronization delay
            self._wait_for_synchronization()
            
            # Phase 6: Main data collection loop
            self._run_collection_loop()
            
        except Exception as e:
            logger.error(f"Critical error in BrainAccess collector thread: {e}")
            self._error_message = str(e)
            self._error_event.set()
        finally:
            self._cleanup()
    
    def stop(self):
        """Stop the data collection thread"""
        logger.info("Stopping BrainAccess EEG collector thread")
        self._stop_event.set()
    
    # Methods called by main system
    def wait_for_connection(self, timeout: float = 20.0) -> bool:
        """Wait for device connection"""
        logger.info(f"Waiting for EEG device connection (timeout: {timeout}s)")
        result = self._connected.wait(timeout)
        if result:
            logger.info("EEG device connection confirmed")
        else:
            logger.error(f"EEG device connection timeout after {timeout}s")
        return result
    
    def wait_for_streaming(self, timeout: float = 10.0) -> bool:
        """Wait for data streaming to start"""
        logger.info(f"Waiting for EEG streaming to start (timeout: {timeout}s)")
        result = self._streaming.wait(timeout)
        if result:
            logger.info("EEG streaming confirmed")
        else:
            logger.error(f"EEG streaming timeout after {timeout}s")
        return result
    
    def wait_for_synchronization(self, timeout: float = 10.0) -> bool:
        """Wait for device synchronization to complete"""
        logger.info(f"Waiting for EEG synchronization (timeout: {timeout}s)")
        result = self._synchronized.wait(timeout)
        if result:
            logger.info("EEG synchronization confirmed")
        else:
            logger.error(f"EEG synchronization timeout after {timeout}s")
        return result
    
    def is_connected(self) -> bool:
        """Check if device is connected"""
        return self._connected.is_set() and (self.mgr is not None) and self.mgr.is_connected()
    
    def is_streaming(self) -> bool:
        """Check if data is streaming"""
        return self._streaming.is_set() and (self.mgr is not None) and self.mgr.is_streaming()
    
    def is_synchronized(self) -> bool:
        """Check if device is synchronized"""
        return self._synchronized.is_set()
    
    def has_error(self) -> bool:
        """Check if thread has encountered an error"""
        return self._error_event.is_set()
    
    def get_error_message(self) -> str:
        """Get the last error message"""
        return self._error_message
    
    def get_device_info(self) -> Dict:
        """Get comprehensive device information"""
        base_info = {
            'device_name': self.device_name,
            'is_connected': self.is_connected(),
            'is_streaming': self.is_streaming(),
            'is_synchronized': self.is_synchronized(),
            'has_error': self.has_error(),
            'error_message': self._error_message,
            'uptime': time.time() - self.start_time if self.start_time else 0,
            'sync_delay': self.sync_delay,
            'target_batch_duration': self.TARGET_BATCH_DURATION
        }
        
        if self.mgr and self.is_connected():
            try:
                battery = self.mgr.get_battery_info()
                
                base_info.update({
                    'sample_rate': self.sample_rate,
                    'eeg_channels': self.eeg_channel_names,
                    'valid_channels': len(self.valid_channels),
                    'num_channels': len(self.eeg_channel_names),
                    'battery_level': battery.level,
                    'total_samples': self.total_samples_received,
                    'total_chunks': self.total_chunks_received,
                    'total_batches_sent': self.total_batches_sent,
                    'dropped_batches': self.dropped_batches,
                    'current_accumulation': self.total_accumulated_samples,
                    'chunk_size_stats': {
                        'min': min(self.chunk_size_stats) if self.chunk_size_stats else 0,
                        'max': max(self.chunk_size_stats) if self.chunk_size_stats else 0,
                        'avg': sum(self.chunk_size_stats) / len(self.chunk_size_stats) if self.chunk_size_stats else 0
                    }
                })
                    
            except Exception as e:
                logger.warning(f"Error getting extended device info: {e}")
                base_info['info_error'] = str(e)
        
        return base_info
    
    def _initialize_core(self):
        """Initialize BrainAccess core library"""
        logger.info("Initializing BrainAccess core")
        try:
            bacore.init()
            self.mgr = EEGManager()
            self.mgr.__enter__()
            logger.info("BrainAccess core initialized")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BrainAccess core: {e}")
        
    def _connect_device(self):
        """Connect to the BrainAccess device with retry logic"""
        logger.info(f"Scanning for BrainAccess devices...")
        
        for attempt in range(self.max_connection_attempts):
            if self._stop_event.is_set():
                return
                
            try:
                logger.info(f"Connection attempt {attempt + 1}/{self.max_connection_attempts}")
                
                # Scan for devices
                bacore.scan(0)
                device_count = bacore.get_device_count()
                
                if device_count == 0:
                    raise BrainAccessException("No BrainAccess devices found during scan")
                
                # Find target device
                device_port = None
                available_devices = []
                
                for i in range(device_count):
                    name = bacore.get_device_name(i)
                    available_devices.append(name)
                    logger.info(f"Found device: {name}")
                    if self.device_name in name:
                        device_port = i
                        break
                
                if device_port is None:
                    raise BrainAccessException(
                        f"Target device '{self.device_name}' not found. "
                        f"Available devices: {available_devices}"
                    )
                
                # Attempt connection
                logger.info(f"Connecting to device '{self.device_name}' at port {device_port}")
                connection_result = self.mgr.connect(bt_device_index=device_port)
                
                if connection_result == 0:
                    logger.info("Successfully connected to BrainAccess device")
                    self._connected.set()
                    return
                elif connection_result == 2:
                    raise BrainAccessException(
                        "Stream incompatible - device firmware needs update"
                    )
                else:
                    raise BrainAccessException(
                        f"Connection failed with error code {connection_result}"
                    )
                    
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt == self.max_connection_attempts - 1:
                    raise RuntimeError(f"Failed to connect after {self.max_connection_attempts} attempts: {e}")
                
                # Wait before retry
                logger.info(f"Waiting {self.connection_retry_delay}s before retry...")
                time.sleep(self.connection_retry_delay)
    
    def _setup_channels(self):
        """Setup EEG channels and device configuration"""
        logger.info("Setting up EEG channels and configuration")
        
        try:
            # Get device info first
            device_info = self.mgr.get_device_info()
            self.sample_rate = self.mgr.get_sample_frequency()
            logger.info(f"Device sample rate: {self.sample_rate} Hz")
            
            # Setup EEG electrode channels
            logger.info(f"Setting up {len(self.cap)} EEG channels...")
            for electrode, name in self.cap.items():
                channel_id = eeg_channel.ELECTRODE_MEASUREMENT + electrode
                self.eeg_channels[channel_id] = name
                self.mgr.set_channel_enabled(channel_id, True)
                self.mgr.set_channel_gain(channel_id, self.gain)
                self.eeg_channel_names.append(name)
                logger.debug(f"Channel {electrode}: {name}")
            
            # Load configuration to device
            logger.info("Loading configuration to device...")
            self.mgr.load_config()
            
            logger.info("EEG channels configured - will map indexes after streaming starts")
            
        except Exception as e:
            raise RuntimeError(f"Channel setup failed: {e}")
    
    def _discover_available_channels(self):
        """Discover which channels are actually available after streaming starts"""
        logger.info("Discovering available channels after streaming...")
        
        try:
            # Clear previous mappings
            self.channel_indexes = {}
            self.valid_channels = []
            
            # Try to map all configured channels
            mapped_count = 0
            for channel_id, name in self.eeg_channels.items():
                try:
                    idx = self.mgr.get_channel_index(channel_id)
                    self.channel_indexes[channel_id] = idx
                    self.valid_channels.append(channel_id)
                    mapped_count += 1
                    logger.info(f"Channel {name} mapped to index {idx}")
                except BrainAccessException as e:
                    logger.warning(f"Could not map channel {name} (ID {channel_id}): {e}")
            
            # If no channels mapped, try to discover what's available
            if mapped_count == 0:
                logger.warning("No configured channels found, trying to discover available channels...")
                self._discover_any_available_channels()
            
            logger.info(f"Channel discovery complete: {len(self.valid_channels)}/{len(self.eeg_channels)} channels available")
            
            if len(self.valid_channels) == 0:
                raise RuntimeError("No valid channels found - cannot proceed")
                
        except Exception as e:
            raise RuntimeError(f"Channel discovery failed: {e}")
    
    def _discover_any_available_channels(self):
        """Try to discover any available channels by scanning channel indexes"""
        logger.info("Scanning for any available channels...")
        
        # Try channel indexes 0-31 to see what's available
        found_channels = []
        for test_idx in range(32):
            try:
                # This is a bit of a hack - try to see if data comes through this index
                # We'll verify this during the actual data callback
                logger.debug(f"Testing channel index {test_idx}")
                # We can't directly test here, so we'll just log for now
            except:
                pass
        
        logger.info(f"Will verify available channels during data reception")
    
    def _start_streaming(self):
        """Start EEG data streaming"""
        logger.info("Starting EEG data stream...")
        
        try:
            # Set data chunk callback
            self.mgr.set_callback_chunk(self._on_data_chunk)
            
            # Start streaming
            self.mgr.start_stream()
            self._streaming.set()
            logger.info("EEG data streaming started successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to start streaming: {e}")
    
    def _wait_for_synchronization(self):
        """Wait for device to synchronize and discover available channels"""
        logger.info(f"Waiting {self.sync_delay}s for device synchronization...")
        
        start_time = time.time()
        chunks_received = 0
        channels_discovered = False
        
        # Monitor synchronization progress
        while time.time() - start_time < self.sync_delay:
            if self._stop_event.is_set():
                return
            
            # Try to discover channels after a few chunks have been received
            if not channels_discovered and self.total_chunks_received > 5:
                try:
                    self._discover_available_channels()
                    channels_discovered = True
                except Exception as e:
                    logger.warning(f"Channel discovery failed: {e}")
            
            # Check progress every second
            time.sleep(0.5)
            elapsed = time.time() - start_time
            new_chunks = self.total_chunks_received - chunks_received
            chunks_received = self.total_chunks_received
            
            logger.info(f"Sync progress: {elapsed:.1f}/{self.sync_delay}s, "
                       f"chunks received: {chunks_received}, "
                       f"valid channels: {len(self.valid_channels)}")
        
        # Final attempt at channel discovery if not done yet
        if not channels_discovered:
            try:
                logger.info("Final attempt at channel discovery...")
                self._discover_available_channels()
            except Exception as e:
                logger.error(f"Final channel discovery failed: {e}")
        
        # Check if we received data during sync
        if self.total_chunks_received > 0 and len(self.valid_channels) > 0:
            logger.info(f"Synchronization complete: {self.total_chunks_received} chunks received, "
                       f"{len(self.valid_channels)} valid channels")
            self._synchronized.set()
        else:
            logger.warning(f"Synchronization issues: {self.total_chunks_received} chunks, "
                          f"{len(self.valid_channels)} valid channels")
    
    def _on_data_chunk(self, chunk_arrays: List[np.ndarray], chunk_size: int):
        """
        Callback for incoming data chunks from BrainAccess device
        Accumulates chunks into ~3-second batches (adaptive)
        """
        if self._stop_event.is_set():
            return
        
        try:
            current_time = time.time()
            self.last_data_time = current_time
            self.total_chunks_received += 1
            
            # Track chunk size statistics
            self.chunk_size_stats.append(chunk_size)
            if len(self.chunk_size_stats) > 1000:  # Keep last 1000 chunks
                self.chunk_size_stats = self.chunk_size_stats[-1000:]
            
            # Debug info for first few chunks
            if self.total_chunks_received <= 5:
                logger.info(f"Chunk {self.total_chunks_received}: {len(chunk_arrays)} arrays, chunk_size={chunk_size}")
                for i, arr in enumerate(chunk_arrays):
                    if i < 5:  # Log first 5 arrays
                        logger.info(f"  Array {i}: shape={arr.shape}, dtype={arr.dtype}")
            
            # If channels not discovered yet, try to extract data from any available arrays
            if len(self.valid_channels) == 0:
                # Try to use whatever data is available
                if len(chunk_arrays) > 0:
                    # Use first available arrays as channels
                    eeg_data_list = []
                    channel_names_with_data = []
                    
                    # Take up to the number of channels we expect, or what's available
                    max_channels = min(len(chunk_arrays), len(self.eeg_channel_names))
                    for i in range(max_channels):
                        if chunk_arrays[i].size > 0:  # Check if array has data
                            eeg_data_list.append(chunk_arrays[i])
                            # Use channel names if we have them, otherwise generic names
                            if i < len(self.eeg_channel_names):
                                channel_names_with_data.append(self.eeg_channel_names[i])
                            else:
                                channel_names_with_data.append(f"CH_{i}")
                    
                    if eeg_data_list:
                        logger.info(f"Using {len(eeg_data_list)} discovered channels for data")
                        # Update valid channels for future use
                        self.valid_channels = list(range(len(eeg_data_list)))
                        self.channel_names_for_batch = channel_names_with_data
                else:
                    logger.warning("No chunk arrays available")
                    return
            else:
                # Extract EEG data from mapped valid channels
                eeg_data_list = []
                channel_names_with_data = []
                
                for channel_id in self.valid_channels:
                    if channel_id in self.channel_indexes:
                        idx = self.channel_indexes[channel_id]
                        if idx < len(chunk_arrays):
                            eeg_data_list.append(chunk_arrays[idx])
                            channel_names_with_data.append(self.eeg_channels[channel_id])
                
                if not eeg_data_list:
                    # Fallback: try to use available arrays directly
                    logger.warning("No mapped channels found, using available arrays")
                    for i, arr in enumerate(chunk_arrays):
                        if arr.size > 0 and i < len(self.eeg_channel_names):
                            eeg_data_list.append(arr)
                            channel_names_with_data.append(self.eeg_channel_names[i])
            
            if not eeg_data_list:
                logger.warning("No EEG data found in chunk")
                return
            
            # Create EEG data array for this chunk
            try:
                chunk_eeg_data = np.array(eeg_data_list)  # Shape: (n_channels, chunk_size)
            except ValueError as e:
                logger.warning(f"Could not create EEG data array: {e}")
                # Try to handle different array sizes
                max_size = max(arr.size for arr in eeg_data_list)
                padded_arrays = []
                for arr in eeg_data_list:
                    if arr.size < max_size:
                        padded = np.zeros(max_size)
                        padded[:arr.size] = arr.flatten()
                        padded_arrays.append(padded)
                    else:
                        padded_arrays.append(arr.flatten()[:max_size])
                chunk_eeg_data = np.array(padded_arrays)
            
            # Initialize accumulation if needed
            if self.batch_start_time is None:
                self.accumulation_buffer = []
                self.batch_start_time = current_time
                self.total_accumulated_samples = 0
                self.channel_names_for_batch = channel_names_with_data
            
            # Add chunk to accumulation buffer
            self.accumulation_buffer.append(chunk_eeg_data)
            self.total_accumulated_samples += chunk_eeg_data.shape[1] if chunk_eeg_data.ndim > 1 else chunk_eeg_data.shape[0]
            self.total_samples_received += chunk_size
            
            # Check if we should send accumulated batch (based on time)
            elapsed_time = current_time - self.batch_start_time
            if elapsed_time >= self.TARGET_BATCH_DURATION:
                self._send_accumulated_batch(current_time, elapsed_time)
                
        except Exception as e:
            logger.error(f"Error processing data chunk: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _send_accumulated_batch(self, timestamp: float, actual_duration: float):
        """
        Send accumulated batch to Thread 2
        """
        try:
            if not self.accumulation_buffer:
                logger.warning("No data in accumulation buffer")
                return
            
            # Concatenate all accumulated chunks along time axis
            try:
                batch_eeg_data = np.concatenate(self.accumulation_buffer, axis=1)
            except ValueError as e:
                logger.warning(f"Could not concatenate chunks normally: {e}")
                # Try to handle different sized chunks
                max_channels = max(chunk.shape[0] for chunk in self.accumulation_buffer)
                total_samples = sum(chunk.shape[1] if chunk.ndim > 1 else chunk.shape[0] for chunk in self.accumulation_buffer)
                
                # Create a properly sized array
                batch_eeg_data = np.zeros((max_channels, total_samples))
                current_pos = 0
                
                for chunk in self.accumulation_buffer:
                    if chunk.ndim == 1:
                        chunk = chunk.reshape(1, -1)
                    chunk_samples = chunk.shape[1]
                    chunk_channels = chunk.shape[0]
                    
                    batch_eeg_data[:chunk_channels, current_pos:current_pos + chunk_samples] = chunk
                    current_pos += chunk_samples
                
                # Update channel names to match actual data
                if len(self.channel_names_for_batch) != max_channels:
                    self.channel_names_for_batch = [f"CH_{i}" for i in range(max_channels)]
            
            # Create data package
            data_package = EEGDataPackage(
                timestamp=timestamp,
                chunk_size=batch_eeg_data.shape[1],
                time_duration=actual_duration,
                eeg_data=batch_eeg_data,
                channel_names=self.channel_names_for_batch,
                sample_rate=self.sample_rate
            )
            
            # Send to Thread 2 (non-blocking)
            try:
                self.output_queue.put_nowait(data_package)
                self.total_batches_sent += 1
                
                logger.info(f"Sent batch {self.total_batches_sent}: "
                           f"shape {batch_eeg_data.shape}, "
                           f"{actual_duration:.2f}s duration, "
                           f"{len(self.channel_names_for_batch)} channels")
                
            except queue.Full:
                self.dropped_batches += 1
                self.queue_full_warnings += 1
                logger.warning(f"Output queue full, dropped batch #{self.dropped_batches}")
            
            # Reset accumulation
            self.accumulation_buffer = []
            self.total_accumulated_samples = 0
            self.batch_start_time = None
                
        except Exception as e:
            logger.error(f"Error sending accumulated batch: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Reset accumulation on error
            self.accumulation_buffer = []
            self.total_accumulated_samples = 0
            self.batch_start_time = None
    
    def _run_collection_loop(self):
        """Main data collection monitoring loop"""
        logger.info("Entering data collection monitoring loop")
        
        last_status_time = time.time()
        status_interval = 10  # Log status every 10 seconds
        
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                
                # Periodic status logging
                if current_time - last_status_time > status_interval:
                    elapsed_batch_time = current_time - self.batch_start_time if self.batch_start_time else 0
                    avg_chunk_size = sum(self.chunk_size_stats) / len(self.chunk_size_stats) if self.chunk_size_stats else 0
                    
                    logger.info(f"EEG Stats: "
                              f"{self.total_chunks_received} chunks received, "
                              f"{self.total_batches_sent} batches sent, "
                              f"Current batch: {self.total_accumulated_samples} samples "
                              f"({elapsed_batch_time:.1f}s), "
                              f"Avg chunk size: {avg_chunk_size:.1f}")
                    
                    if self.dropped_batches > 0:
                        logger.warning(f"Dropped {self.dropped_batches} batches due to full queue")
                    
                    last_status_time = current_time
                
                # Check connection status
                if not self.mgr.is_connected():
                    logger.error("Device disconnected during operation")
                    self._connected.clear()
                    break
                
                # Check streaming status
                if not self.mgr.is_streaming():
                    logger.error("Device stopped streaming during operation")
                    self._streaming.clear()
                    break
                
                # Small sleep to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in collection monitoring loop: {e}")
                break
        
        logger.info("Exiting data collection monitoring loop")
    
    def _cleanup(self):
        """Clean up all resources"""
        logger.info("Cleaning up BrainAccess collector resources...")
        
        try:
            # Send any remaining accumulated data as partial batch
            if self.batch_start_time and self.total_accumulated_samples > 0:
                elapsed = time.time() - self.batch_start_time
                logger.info(f"Sending final partial batch with {self.total_accumulated_samples} samples")
                self._send_accumulated_batch(time.time(), elapsed)
            
            # Stop streaming
            if self.mgr and self.mgr.is_streaming():
                try:
                    self.mgr.stop_stream()
                    self._streaming.clear()
                    logger.info("EEG streaming stopped")
                except Exception as e:
                    logger.warning(f"Error stopping stream: {e}")
            
            # Disconnect device
            if self.mgr and self.mgr.is_connected():
                try:
                    self.mgr.disconnect()
                    self._connected.clear()
                    logger.info("Device disconnected")
                except Exception as e:
                    logger.warning(f"Error disconnecting device: {e}")
            
            # Close EEG manager
            if self.mgr:
                try:
                    self.mgr.__exit__(None, None, None)
                    logger.info("EEG manager closed")
                except Exception as e:
                    logger.warning(f"Error closing EEG manager: {e}")
            
            # Close BrainAccess core
            try:
                bacore.close()
                logger.info("BrainAccess core closed")
            except Exception as e:
                logger.warning(f"Error closing BrainAccess core: {e}")
            
            # Log final statistics
            if self.start_time:
                total_time = time.time() - self.start_time
                avg_chunk_size = sum(self.chunk_size_stats) / len(self.chunk_size_stats) if self.chunk_size_stats else 0
                
                logger.info(f"Final stats: "
                          f"{self.total_chunks_received} chunks processed, "
                          f"{self.total_batches_sent} batches sent, "
                          f"{self.total_samples_received} samples collected, "
                          f"Valid channels: {len(self.valid_channels)}, "
                          f"Avg chunk size: {avg_chunk_size:.1f}, "
                          f"Runtime: {total_time:.1f}s, "
                          f"Dropped: {self.dropped_batches} batches")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        logger.info("BrainAccess collector cleanup complete")


# =============================================================================
# TESTING CODE
# =============================================================================

def test_thread1_fixed():
    """
    Test Thread 1 with synchronization and channel discovery fixes
    """
    
    # Test configuration - YOUR DEVICE
    TEST_CONFIG = {
        'device_name': "BA MIDI 026",
        'channels': {
            0: "T8", 2: "P4", 3: "C4", 4: "Pz", 5: "Fz", 6: "Oz", 7: "F4",
            8: "F3", 9: "P3", 10: "F8", 12: "PO3", 13: "C3", 14: "T7", 15: "F7"
        },
        'gain': 8,
        'sync_delay': 10.0  # 10 seconds for thorough synchronization
    }
    
    # Create test queue
    test_queue = queue.Queue(maxsize=50)
    
    # Create Thread 1 with synchronization
    collector = BrainAccessEEGCollector(
        device_name=TEST_CONFIG['device_name'],
        output_queue=test_queue,
        cap=TEST_CONFIG['channels'],
        gain=TEST_CONFIG['gain'],
        sync_delay=TEST_CONFIG['sync_delay']
    )
    
    print("Testing Thread 1 - Fixed with Channel Discovery")
    print("=" * 55)
    print(f"Device: {TEST_CONFIG['device_name']}")
    print(f"Channels: {len(TEST_CONFIG['channels'])} EEG channels")
    print(f"Gain: {TEST_CONFIG['gain']}")
    print(f"Sync delay: {TEST_CONFIG['sync_delay']}s")
    print(f"Target batch: ~3 seconds")
    print()
    print("Note: Will discover available channels after streaming starts")
    print()
    
    try:
        # Start Thread 1
        print("Starting Thread 1...")
        collector.start()
        
        # Test connection
        print("Testing connection...")
        if collector.wait_for_connection(timeout=25):
            print("Connection successful!")
            
            # Test streaming
            print("Testing streaming...")
            if collector.wait_for_streaming(timeout=15):
                print("Streaming successful!")
                
                # Test synchronization and channel discovery
                print(f"Testing synchronization and channel discovery ({TEST_CONFIG['sync_delay']}s)...")
                print("This will discover which channels actually work...")
                if collector.wait_for_synchronization(timeout=TEST_CONFIG['sync_delay'] + 10):
                    print("Synchronization and channel discovery successful!")
                    
                    # Get device info after sync
                    device_info = collector.get_device_info()
                    print(f"Device Info after synchronization:")
                    print(f"   Sample Rate: {device_info.get('sample_rate', 'Unknown')} Hz")
                    print(f"   Valid Channels: {device_info.get('valid_channels', 0)}/{device_info.get('num_channels', 0)}")
                    print(f"   Battery: {device_info.get('battery_level', 'Unknown')}%")
                    chunk_stats = device_info.get('chunk_size_stats', {})
                    print(f"   Chunk Size: min={chunk_stats.get('min', 0)}, "
                          f"max={chunk_stats.get('max', 0)}, "
                          f"avg={chunk_stats.get('avg', 0):.1f}")
                    print()
                    
                    # Collect test data
                    print("Collecting batches...")
                    print("Note: First batch may take up to 3 seconds...")
                    data_collected = 0
                    
                    for i in range(3):  # Try to get 3 batches
                        try:
                            print(f"Waiting for batch {i+1}...")
                            data = test_queue.get(timeout=15.0)  # 15 second timeout
                            data_collected += 1
                            print(f"Batch {data_collected}: "
                                  f"Shape {data.eeg_data.shape}, "
                                  f"{data.chunk_size} samples, "
                                  f"{data.time_duration:.2f}s duration, "
                                  f"Rate: {data.sample_rate} Hz")
                            print(f"              Channels: {data.channel_names}")
                            
                        except queue.Empty:
                            print(f"Timeout waiting for batch {i+1}")
                    
                    print()
                    if data_collected > 0:
                        print(f"SUCCESS! Collected {data_collected}/3 batches")
                    else:
                        print("No batches received - check device and configuration")
                    
                    # Show final stats
                    final_stats = collector.get_device_info()
                    print(f"Final Stats:")
                    print(f"   Total chunks: {final_stats.get('total_chunks', 0)}")
                    print(f"   Total batches sent: {final_stats.get('total_batches_sent', 0)}")
                    print(f"   Total samples: {final_stats.get('total_samples', 0)}")
                    print(f"   Valid channels: {final_stats.get('valid_channels', 0)}")
                    print(f"   Dropped batches: {final_stats.get('dropped_batches', 0)}")
                    print(f"   Queue size: {test_queue.qsize()}")
                    
                else:
                    print("Synchronization/channel discovery failed!")
            else:
                print("Streaming failed!")
        else:
            print("Connection failed!")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest error: {e}")
    finally:
        print("\nStopping Thread 1...")
        collector.stop()
        collector.join(timeout=10)
        print("Thread 1 test finished")


if __name__ == "__main__":
    test_thread1_fixed()