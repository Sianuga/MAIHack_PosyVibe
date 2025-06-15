"""
thread2_eeg_processor.py - EEG Filtering and Downsampling
========================================================

Thread 2: EEG Data Processing
- Receives ~3-second batches from Thread 1
- Applies bandpass + notch filtering using filter.py
- Downsamples from 250 Hz to 1 Hz using RMS averaging
- Outputs DataFrame format: time column + channel columns
- Sends 3 rows per batch (1 row per second)
"""

import threading
import queue
import time
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging
from dataclasses import dataclass

# Import the filter function
from filter import filter_eeg_like_mne
from thread1_eeg_connector import EEGDataPackage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessedEEGData:
    """
    Processed EEG data package sent from Thread 2 to Thread 3
    
    Format: DataFrame with columns [time, CH1, CH2, ..., CHn]
    - time: relative seconds from session start
    - CHx: RMS-averaged filtered EEG data (1 sample per second)
    - Always contains exactly 3 rows (representing 3 seconds)
    """
    timestamp: float              # System timestamp when processing completed
    batch_id: int                # Sequential batch ID
    session_time_start: float    # Session time of first sample (relative seconds)
    session_time_end: float      # Session time of last sample (relative seconds)
    data_frame: pd.DataFrame     # Processed data: [time, CH1, CH2, ..., CHn]
    original_duration: float     # Original batch duration before processing
    original_samples: int        # Original number of samples before downsampling
    channel_names: List[str]     # Channel names in order
    sample_rate_original: int    # Original sample rate
    sample_rate_output: int      # Output sample rate (always 1 Hz)

class EEGFilterProcessor(threading.Thread):
    """
    Thread 2: EEG Filtering and Downsampling
    
    Processing Pipeline:
    1. Receive 3-second batches from Thread 1
    2. Apply bandpass + notch filtering (0.5-40 Hz + 60 Hz notch)
    3. Downsample to 1 Hz using RMS averaging
    4. Format as DataFrame with time column
    5. Send to Thread 3
    """
    
    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        filter_config: Optional[Dict] = None
    ):
        """
        Initialize EEG filter processor thread
        
        Parameters:
        -----------
        input_queue : queue.Queue
            Queue to receive EEGDataPackage objects from Thread 1
        output_queue : queue.Queue
            Queue to send ProcessedEEGData objects to Thread 3
        filter_config : Dict, optional
            Filter configuration. If None, uses defaults:
            {
                'l_freq': 0.5,      # High-pass frequency (Hz)
                'h_freq': 40,       # Low-pass frequency (Hz) 
                'notch_freq': 60    # Notch filter frequency (Hz)
            }
        """
        super().__init__(name="EEGFilterProcessor", daemon=True)
        
        self.input_queue = input_queue
        self.output_queue = output_queue
        
        # Filter configuration - adjusted for short segments
        self.filter_config = filter_config or {
            'l_freq': 1.0,      # High-pass: 1Hz for short segments (was 0.5Hz)
            'h_freq': 40,       # Low-pass: remove high-frequency noise, keep gamma
            'notch_freq': 60    # Notch: remove power line interference
        }
        
        # Thread control
        self._stop_event = threading.Event()
        self._processing_active = threading.Event()
        
        # Session tracking
        self.session_start_time = None  # When first batch was processed
        self.batch_counter = 0
        self.total_samples_processed = 0
        self.total_batches_processed = 0
        
        # Processing statistics
        self.start_time = None
        self.last_process_time = None
        self.processing_times = []  # Track processing duration per batch
        self.dropped_batches = 0
        self.filter_errors = 0
        
        # Output configuration
        self.output_sample_rate = 2  # 1 Hz output
        self.samples_per_output = 2  # Always output 3 samples (3 seconds)
        
        logger.info(f"EEG Filter Processor initialized")
        logger.info(f"Filter config: {self.filter_config}")
        logger.info(f"Output: {self.samples_per_output} samples at {self.output_sample_rate} Hz")
    
    def run(self):
        """Main thread execution"""
        logger.info("Starting EEG Filter Processor thread")
        self.start_time = time.time()
        self._processing_active.set()
        
        try:
            self._run_processing_loop()
        except Exception as e:
            logger.error(f"Critical error in EEG Filter Processor: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self._cleanup()
    
    def stop(self):
        """Stop the processing thread"""
        logger.info("Stopping EEG Filter Processor")
        self._stop_event.set()
    
    def is_active(self) -> bool:
        """Check if thread is actively processing"""
        return self._processing_active.is_set()
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        stats = {
            'is_active': self.is_active(),
            'session_start_time': self.session_start_time,
            'total_batches_processed': self.total_batches_processed,
            'total_samples_processed': self.total_samples_processed,
            'dropped_batches': self.dropped_batches,
            'filter_errors': self.filter_errors,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'filter_config': self.filter_config,
            'uptime': time.time() - self.start_time if self.start_time else 0
        }
        
        if self.processing_times:
            stats['processing_performance'] = {
                'avg_processing_time': np.mean(self.processing_times),
                'max_processing_time': np.max(self.processing_times),
                'min_processing_time': np.min(self.processing_times),
                'last_processing_time': self.processing_times[-1] if self.processing_times else 0
            }
        
        return stats
    
    def _run_processing_loop(self):
        """Main processing loop"""
        logger.info("Entering EEG processing loop")
        
        last_status_time = time.time()
        status_interval = 15  # Log status every 15 seconds
        
        while not self._stop_event.is_set():
            try:
                # Wait for input data with timeout
                try:
                    raw_batch = self.input_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the batch
                process_start_time = time.time()
                processed_data = self._process_batch(raw_batch)
                process_duration = time.time() - process_start_time
                
                self.processing_times.append(process_duration)
                if len(self.processing_times) > 100:  # Keep last 100 measurements
                    self.processing_times = self.processing_times[-100:]
                
                if processed_data is not None:
                    # Send to Thread 3
                    try:
                        self.output_queue.put_nowait(processed_data)
                        logger.debug(f"Sent processed batch {processed_data.batch_id} "
                                   f"(processing time: {process_duration:.3f}s)")
                    except queue.Full:
                        self.dropped_batches += 1
                        logger.warning(f"Output queue full, dropped processed batch "
                                     f"{processed_data.batch_id}")
                
                # Periodic status logging
                current_time = time.time()
                if current_time - last_status_time > status_interval:
                    self._log_status()
                    last_status_time = current_time
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info("Exiting EEG processing loop")
    
    def _process_batch(self, raw_batch: EEGDataPackage) -> Optional[ProcessedEEGData]:
        """
        Process a single batch: filter + downsample + format
        
        Parameters:
        -----------
        raw_batch : EEGDataPackage
            Raw EEG data from Thread 1
            
        Returns:
        --------
        ProcessedEEGData or None if processing failed
        """
        try:
            batch_start_time = time.time()
            self.batch_counter += 1
            
            # Initialize session timing on first batch
            if self.session_start_time is None:
                self.session_start_time = raw_batch.timestamp
                logger.info(f"Session started at timestamp {self.session_start_time}")
            
            # Calculate session time for this batch
            batch_session_time = raw_batch.timestamp - self.session_start_time
            
            logger.debug(f"Processing batch {self.batch_counter}: "
                        f"shape {raw_batch.eeg_data.shape}, "
                        f"duration {raw_batch.time_duration:.2f}s, "
                        f"session time {batch_session_time:.2f}s")
            
            # Step 1: Apply filtering
            filtered_data = self._apply_filtering(raw_batch)
            if filtered_data is None:
                return None
            
            # Step 2: Downsample to 1 Hz using RMS averaging
            downsampled_data = self._downsample_to_1hz(
                filtered_data, 
                raw_batch.sample_rate,
                raw_batch.time_duration
            )
            
            # Step 3: Create time column (relative to session start)
            time_column = self._create_time_column(
                batch_session_time, 
                raw_batch.time_duration
            )
            
            # Step 4: Format as DataFrame
            data_frame = self._format_as_dataframe(
                downsampled_data,
                time_column,
                raw_batch.channel_names
            )
            
            # Step 5: Create processed data package
            session_time_start = batch_session_time
            session_time_end = batch_session_time + raw_batch.time_duration
            
            processed_data = ProcessedEEGData(
                timestamp=time.time(),
                batch_id=self.batch_counter,
                session_time_start=session_time_start,
                session_time_end=session_time_end,
                data_frame=data_frame,
                original_duration=raw_batch.time_duration,
                original_samples=raw_batch.chunk_size,
                channel_names=raw_batch.channel_names,
                sample_rate_original=raw_batch.sample_rate,
                sample_rate_output=self.output_sample_rate
            )
            
            self.total_batches_processed += 1
            self.total_samples_processed += raw_batch.chunk_size
            self.last_process_time = time.time()
            
            processing_duration = time.time() - batch_start_time
            logger.debug(f"Batch {self.batch_counter} processed in {processing_duration:.3f}s")
            
            return processed_data
            
        except Exception as e:
            self.filter_errors += 1
            logger.error(f"Error processing batch {self.batch_counter}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _apply_filtering(self, raw_batch: EEGDataPackage) -> Optional[np.ndarray]:
        """
        Apply filtering with fallback for short segments
        
        For 3-second segments, uses simpler Butterworth filters instead of long FIR filters
        """
        try:
            n_samples = raw_batch.eeg_data.shape[1]
            
            # For short segments (< 1000 samples), use simple Butterworth filters
            if n_samples < 1000:
                logger.debug(f"Using Butterworth filters for short segment ({n_samples} samples)")
                filtered_data = self._apply_butterworth_filtering(raw_batch)
            else:
                # For longer segments, try the original MNE-style filtering with adjustments
                adjusted_config = self._adjust_filter_for_data_length(
                    self.filter_config.copy(), 
                    n_samples, 
                    raw_batch.sample_rate
                )
                
                filtered_data = filter_eeg_like_mne(
                    data=raw_batch.eeg_data,
                    sfreq=raw_batch.sample_rate,
                    l_freq=adjusted_config['l_freq'],
                    h_freq=adjusted_config['h_freq'],
                    notch_freq=adjusted_config['notch_freq']
                )
            
            logger.debug(f"Filtering applied: {raw_batch.eeg_data.shape} -> {filtered_data.shape}")
            return filtered_data
            
        except Exception as e:
            logger.error(f"Filtering failed: {e}")
            # Final fallback: return unfiltered data with warning
            logger.warning("Returning unfiltered data as fallback")
            return raw_batch.eeg_data.copy()
    
    def _apply_butterworth_filtering(self, raw_batch: EEGDataPackage) -> np.ndarray:
        """
        Apply simple Butterworth filtering for short EEG segments
        
        Uses scipy.signal.butter + filtfilt with much shorter filter lengths
        Perfect for 3-second emotion detection segments
        """
        from scipy import signal
        
        data = raw_batch.eeg_data.copy()
        sample_rate = raw_batch.sample_rate
        nyquist = sample_rate / 2.0
        
        # Apply bandpass filter (1-40 Hz) using Butterworth
        if self.filter_config.get('l_freq') and self.filter_config.get('h_freq'):
            low_freq = max(self.filter_config['l_freq'], 0.5)  # At least 0.5 Hz
            high_freq = min(self.filter_config['h_freq'], nyquist - 1)
            
            # 4th order Butterworth bandpass
            sos = signal.butter(
                N=4, 
                Wn=[low_freq, high_freq], 
                btype='band', 
                fs=sample_rate, 
                output='sos'
            )
            
            # Apply to each channel
            for ch_idx in range(data.shape[0]):
                data[ch_idx, :] = signal.sosfiltfilt(sos, data[ch_idx, :])
            
            logger.debug(f"Applied Butterworth bandpass: {low_freq:.1f}-{high_freq:.1f} Hz")
        
        # Apply notch filter (60 Hz) using simple notch
        if self.filter_config.get('notch_freq'):
            notch_freq = self.filter_config['notch_freq']
            if notch_freq < nyquist:
                # Quality factor for notch filter
                Q = 30  # Higher Q = narrower notch
                
                # Design notch filter
                b_notch, a_notch = signal.iirnotch(notch_freq, Q, sample_rate)
                
                # Apply to each channel
                for ch_idx in range(data.shape[0]):
                    data[ch_idx, :] = signal.filtfilt(b_notch, a_notch, data[ch_idx, :])
                
                logger.debug(f"Applied notch filter: {notch_freq} Hz")
        
        return data
    
    def _adjust_filter_for_data_length(
        self, 
        config: Dict, 
        n_samples: int, 
        sample_rate: int
    ) -> Dict:
        """
        Adjust filter parameters to ensure they work with the given data length
        
        For short segments, we need to increase the high-pass frequency to reduce
        the required filter length.
        """
        # Calculate maximum safe high-pass frequency for this data length
        # Rule: filter length should be < n_samples / 6 to allow for padding
        max_safe_filter_length = n_samples // 6
        
        # Calculate what high-pass frequency would give us this filter length
        # filter_length = int(3.3 * sample_rate / l_freq)
        min_safe_l_freq = (3.3 * sample_rate) / max_safe_filter_length
        
        adjusted_config = config.copy()
        
        # Adjust high-pass frequency if necessary
        if config['l_freq'] is not None and config['l_freq'] < min_safe_l_freq:
            adjusted_config['l_freq'] = min_safe_l_freq
            logger.warning(f"Adjusted high-pass frequency from {config['l_freq']:.2f} Hz "
                          f"to {min_safe_l_freq:.2f} Hz for {n_samples} samples")
        
        # For very short segments, disable high-pass filter entirely
        if n_samples < 300:  # Less than ~1.2 seconds at 250 Hz
            adjusted_config['l_freq'] = None
            logger.warning(f"Disabled high-pass filter for very short segment ({n_samples} samples)")
        
        return adjusted_config
    
    def _downsample_to_1hz(
        self, 
        filtered_data: np.ndarray, 
        original_sample_rate: int,
        actual_duration: float
    ) -> np.ndarray:
        """
        Downsample filtered data to 1 Hz using RMS averaging
        
        Always outputs exactly 3 samples regardless of input duration
        
        Parameters:
        -----------
        filtered_data : np.ndarray
            Filtered EEG data, shape (n_channels, n_samples)
        original_sample_rate : int
            Original sample rate (e.g., 250 Hz)
        actual_duration : float
            Actual duration of the batch in seconds
            
        Returns:
        --------
        np.ndarray
            Downsampled data, shape (n_channels, 3)
        """
        n_channels, n_samples = filtered_data.shape
        
        # Always create exactly 3 output samples
        output_samples = self.samples_per_output  # 3
        downsampled = np.zeros((n_channels, output_samples))
        
        # Divide samples into 3 equal-sized windows
        samples_per_window = n_samples // output_samples
        remainder = n_samples % output_samples
        
        logger.debug(f"Downsampling: {n_samples} samples -> {output_samples} samples "
                    f"({samples_per_window} samples per window, {remainder} remainder)")
        
        current_idx = 0
        for window_idx in range(output_samples):
            # Calculate window size (distribute remainder across first windows)
            window_size = samples_per_window + (1 if window_idx < remainder else 0)
            
            if window_size > 0:
                # Extract window data
                window_end = current_idx + window_size
                window_data = filtered_data[:, current_idx:window_end]
                
                # Calculate RMS for each channel in this window
                # RMS = sqrt(mean(x^2)) - better for preserving signal power
                rms_values = np.sqrt(np.mean(window_data ** 2, axis=1))
                downsampled[:, window_idx] = rms_values
                
                current_idx = window_end
            else:
                # Edge case: no samples for this window, use zeros
                downsampled[:, window_idx] = 0.0
                logger.warning(f"Window {window_idx} has no samples")
        
        logger.debug(f"Downsampling complete: output shape {downsampled.shape}")
        
        return downsampled

    def _create_time_column(self, batch_session_time: float, batch_duration: float) -> np.ndarray:
        """
        Create time column with times relative to session/recording start

        Now supports dynamic time points based on batch duration and output rate (e.g., 2Hz = every 0.5s)

        Parameters:
        -----------
        batch_session_time : float
            Time when this batch started (in seconds from session start)
        batch_duration : float
            Duration of the batch (typically 1.0s)

        Returns:
        --------
        np.ndarray
            Time values for each output sample (e.g., [0.0, 0.5] for 1s batch at 2Hz)
        """
        interval = 1.0 / self.output_sample_rate  # 1/2 Hz = 0.5s
        num_samples = int(batch_duration * self.output_sample_rate)  # 1s * 2Hz = 2 próbki

        session_times = np.array([
            batch_session_time + i * interval for i in range(num_samples)
        ])

        logger.debug(f"Time column: batch starts at {batch_session_time:.2f}s, "
                     f"duration: {batch_duration:.1f}s, "
                     f"interval: {interval:.2f}s → times: {session_times}")

        return session_times

    def _format_as_dataframe(
        self, 
        downsampled_data: np.ndarray,
        time_column: np.ndarray,
        channel_names: List[str]
    ) -> pd.DataFrame:
        """
        Format downsampled data as DataFrame with time column
        
        Parameters:
        -----------
        downsampled_data : np.ndarray
            Downsampled EEG data, shape (n_channels, 3)
        time_column : np.ndarray
            Time values for each sample
        channel_names : List[str]
            Names of EEG channels
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: ['time', 'CH1', 'CH2', ..., 'CHn']
        """
        n_channels, n_samples = downsampled_data.shape
        
        # Create DataFrame
        data_dict = {'time': time_column}
        
        # Add channel data
        for ch_idx, ch_name in enumerate(channel_names):
            if ch_idx < n_channels:
                data_dict[ch_name] = downsampled_data[ch_idx, :]
            else:
                logger.warning(f"Channel {ch_name} index {ch_idx} exceeds data shape")
        
        df = pd.DataFrame(data_dict)
        
        logger.debug(f"DataFrame created: {df.shape}, columns: {list(df.columns)}")
        
        return df
    
    def _log_status(self):
        """Log processing status"""
        stats = self.get_statistics()
        
        logger.info(f"Filter Processor Status:")
        logger.info(f"   Batches processed: {stats['total_batches_processed']}")
        logger.info(f"   Samples processed: {stats['total_samples_processed']}")
        logger.info(f"   Input queue: {stats['input_queue_size']}")
        logger.info(f"   Output queue: {stats['output_queue_size']}")
        logger.info(f"   Dropped batches: {stats['dropped_batches']}")
        logger.info(f"   Filter errors: {stats['filter_errors']}")
        
        if 'processing_performance' in stats:
            perf = stats['processing_performance']
            logger.info(f"   Processing time: avg={perf['avg_processing_time']:.3f}s, "
                       f"max={perf['max_processing_time']:.3f}s")
        
        if self.session_start_time:
            session_duration = time.time() - self.session_start_time
            logger.info(f"   Session duration: {session_duration:.1f}s")
    
    def _cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up EEG Filter Processor...")
        
        self._processing_active.clear()
        
        # Log final statistics
        final_stats = self.get_statistics()
        logger.info(f"Final Processing Stats:")
        logger.info(f"   Total batches: {final_stats['total_batches_processed']}")
        logger.info(f"   Total samples: {final_stats['total_samples_processed']}")
        logger.info(f"   Dropped batches: {final_stats['dropped_batches']}")
        logger.info(f"   Filter errors: {final_stats['filter_errors']}")
        logger.info(f"   Uptime: {final_stats['uptime']:.1f}s")
        
        if 'processing_performance' in final_stats:
            perf = final_stats['processing_performance']
            logger.info(f"   Avg processing time: {perf['avg_processing_time']:.3f}s")
        
        logger.info("EEG Filter Processor cleanup complete")


# =============================================================================
# TESTING CODE
# =============================================================================

def test_thread2_processor():
    """
    Test Thread 2 processing with simulated data
    """
    
    # Create test queues
    input_queue = queue.Queue(maxsize=10)
    output_queue = queue.Queue(maxsize=10)
    
    # Test configuration
    filter_config = {
        'l_freq': 1.0,      # High-pass frequency (adjusted for short segments)
        'h_freq': 40,       # Low-pass frequency  
        'notch_freq': 60    # Notch frequency
    }
    
    print("Testing Thread 2 - EEG Filter Processor")
    print("=" * 45)
    print(f"Filter config: {filter_config}")
    print("Expected output: 3 samples per batch at 1 Hz")
    print("Processing: Bandpass + Notch filtering → RMS downsampling")
    print()
    
    # Create Thread 2
    processor = EEGFilterProcessor(
        input_queue=input_queue,
        output_queue=output_queue,
        filter_config=filter_config
    )
    
    try:
        # Start Thread 2
        print("Starting Thread 2...")
        processor.start()
        
        # Create simulated EEG data (similar to Thread 1 output)
        def create_test_batch(batch_id: int, session_time: float) -> EEGDataPackage:
            """Create realistic test EEG data"""
            sample_rate = 250
            duration = 3.0  # 3 seconds
            n_samples = int(sample_rate * duration)  # 750 samples
            n_channels = 14
            
            # Simulate realistic EEG data with different frequency components
            t = np.linspace(0, duration, n_samples)
            eeg_data = np.zeros((n_channels, n_samples))
            
            for ch in range(n_channels):
                # Base signal: mix of alpha (10 Hz), beta (20 Hz), and noise
                alpha = 0.5 * np.sin(2 * np.pi * 10 * t)  # Alpha waves
                beta = 0.3 * np.sin(2 * np.pi * 20 * t)   # Beta waves
                noise = 0.1 * np.random.randn(n_samples)   # Noise
                artifact = 0.2 * np.sin(2 * np.pi * 60 * t)  # 60 Hz artifact (will be filtered)
                
                eeg_data[ch, :] = alpha + beta + noise + artifact
            
            channel_names = [f"CH{i+1}" for i in range(n_channels)]
            
            return EEGDataPackage(
                timestamp=time.time() + session_time,
                chunk_size=n_samples,
                time_duration=duration,
                eeg_data=eeg_data,
                channel_names=channel_names,
                sample_rate=sample_rate
            )
        
        # Send test batches
        print("Sending test batches to Thread 2...")
        test_batches = 3
        
        for i in range(test_batches):
            session_time = i * 3.0  # Each batch starts 3 seconds after previous
            test_batch = create_test_batch(i + 1, session_time)
            
            print(f"Sending batch {i+1}: shape {test_batch.eeg_data.shape}, "
                  f"duration {test_batch.time_duration}s")
            
            input_queue.put(test_batch)
            time.sleep(0.5)  # Small delay
        
        # Collect results
        print("\nWaiting for processed results...")
        results_collected = 0
        
        for i in range(test_batches):
            try:
                processed_data = output_queue.get(timeout=10.0)
                results_collected += 1
                
                df = processed_data.data_frame
                print(f"\nProcessed batch {processed_data.batch_id}:")
                print(f"  Original: {processed_data.original_samples} samples, "
                      f"{processed_data.original_duration:.2f}s")
                print(f"  Output: {df.shape[0]} samples at {processed_data.sample_rate_output} Hz")
                print(f"  Session time: {processed_data.session_time_start:.2f}s - "
                      f"{processed_data.session_time_end:.2f}s")
                print(f"  DataFrame shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                print("  Sample data:")
                print(df.round(4))
                
                # Verify time column
                expected_times = np.array([i*3.0, i*3.0 + 1.0, i*3.0 + 2.0])
                actual_times = df['time'].values
                print(f"  Time verification: expected {expected_times}, got {actual_times}")
                
            except queue.Empty:
                print(f"Timeout waiting for result {i+1}")
        
        # Show processor statistics
        print(f"\nThread 2 Statistics:")
        stats = processor.get_statistics()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")
        
        if results_collected == test_batches:
            print(f"\nSUCCESS! Processed {results_collected}/{test_batches} batches")
        else:
            print(f"\nPARTIAL SUCCESS: {results_collected}/{test_batches} batches processed")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nStopping Thread 2...")
        processor.stop()
        processor.join(timeout=10)
        print("Thread 2 test complete")


if __name__ == "__main__":
    test_thread2_processor()