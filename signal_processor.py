
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_gold_standard(radar_signal, fs):
    """
    Translates the core logic from process_gold_standard.m to Python.
    This function processes a raw radar signal to extract physiological signals.
    """
    try:
        # --- Signal Separation (IIR Filters) ---
        # Low-pass filter for breathing signal
        b_lp, a_lp = butter(4, 0.8, btype='low', fs=fs)
        breathing_signal = filtfilt(b_lp, a_lp, radar_signal)

        # Band-pass filter for heartbeat signal
        b_bp, a_bp = butter(4, [0.9, 2.5], btype='bandpass', fs=fs)
        heartbeat_signal = filtfilt(b_bp, a_bp, radar_signal)

        # --- Feature Extraction ---
        heartbeat_signal_squared = heartbeat_signal ** 2
        prominence_threshold = 1.5 * np.std(heartbeat_signal_squared)
        peaks, _ = find_peaks(
            heartbeat_signal_squared,
            prominence=prominence_threshold,
            distance=fs * 0.5
        )

        extracted_heartbeat_times = peaks / fs
        rr_intervals_raw = np.diff(extracted_heartbeat_times) * 1000  # in ms

        # --- RR Interval Post-Processing ---
        rr_intervals_cleaned = rr_intervals_raw.copy()
        if len(rr_intervals_raw) > 0:
            physio_min_ms = 60000 / 180  # Corresponds to 180 bpm
            physio_max_ms = 60000 / 40   # Corresponds to 40 bpm

            outlier_indices = np.where((rr_intervals_raw < physio_min_ms) | (rr_intervals_raw > physio_max_ms))[0]

            for idx in outlier_indices:
                local_start = max(0, idx - 2)
                local_end = min(len(rr_intervals_cleaned), idx + 3)

                # Get indices of non-outlier neighbors
                local_indices = np.setdiff1d(
                    np.arange(local_start, local_end),
                    outlier_indices
                )

                if len(local_indices) > 0:
                    rr_intervals_cleaned[idx] = np.mean(rr_intervals_cleaned[local_indices])
                else:
                    # Fallback to global mean if no local non-outliers are found
                    global_good_indices = np.setdiff1d(np.arange(len(rr_intervals_cleaned)), outlier_indices)
                    if len(global_good_indices) > 0:
                        rr_intervals_cleaned[idx] = np.mean(rr_intervals_cleaned[global_good_indices])

        # --- HRV Analysis (Welch's method for PSD) ---
        Pxx, f = None, None
        if len(rr_intervals_cleaned) > 10:
            # The tachogram must be uniformly sampled for PSD analysis.
            # We will interpolate it.
            mean_rr_s = np.mean(rr_intervals_cleaned) / 1000.0
            if mean_rr_s > 0:
                fs_tachogram = 1 / mean_rr_s
                # Detrend the signal before PSD
                rr_detrended = rr_intervals_cleaned - np.mean(rr_intervals_cleaned)
                f, Pxx = welch(
                    rr_detrended,
                    fs=fs_tachogram,
                    window='hann',
                    nperseg=len(rr_detrended), # Use the whole signal
                    noverlap=0
                )
            else:
                 logging.warning("Mean RR interval is zero, cannot calculate PSD.")

        return breathing_signal, heartbeat_signal, rr_intervals_cleaned, Pxx, f

    except Exception as e:
        logging.error(f"Error in signal processing: {e}")
        return None, None, None, None, None
