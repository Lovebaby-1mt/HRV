
import os
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import trapezoid
import logging

# Import the core processing function from our existing pipeline
from signal_processor import process_gold_standard

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Configuration ---
INPUT_BASE_DIR = 'sedentary_dataset_py'
OUTPUT_CSV_PATH = 'features_dataset.csv'
CATEGORIES = ['relaxed', 'flow', 'stress', 'drowsiness']

def calculate_features(breathing_signal, rr_intervals, pxx, f, fs):
    """
    Calculates a vector of numerical features from the intermediate signals.
    """
    features = {
        # Time-Domain HRV
        'mean_rr': np.nan,
        'sdnn': np.nan,
        'rmssd': np.nan,
        # Frequency-Domain HRV
        'lf_power': np.nan,
        'hf_power': np.nan,
        'lf_hf_ratio': np.nan,
        # Breathing Features
        'breathing_rate': np.nan,
        'breathing_variance': np.nan,
    }

    # Time-Domain HRV Features (from RR intervals)
    if rr_intervals is not None and len(rr_intervals) > 1:
        features['mean_rr'] = np.mean(rr_intervals)
        features['sdnn'] = np.std(rr_intervals)
        features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals)**2))

    # Frequency-Domain HRV Features (from PSD)
    if pxx is not None and f is not None:
        lf_band = (f >= 0.04) & (f < 0.15)
        hf_band = (f >= 0.15) & (f < 0.4)

        if np.any(lf_band):
            features['lf_power'] = trapezoid(pxx[lf_band], f[lf_band])

        if np.any(hf_band):
            features['hf_power'] = trapezoid(pxx[hf_band], f[hf_band])

        if features['lf_power'] is not np.nan and features['hf_power'] is not np.nan and features['hf_power'] > 0:
            features['lf_hf_ratio'] = features['lf_power'] / features['hf_power']

    # Breathing Features
    if breathing_signal is not None and len(breathing_signal) > fs: # Need at least 1s of data
        features['breathing_variance'] = np.var(breathing_signal)

        # Calculate breathing rate from PSD of the breathing signal
        f_breath, pxx_breath = welch(breathing_signal, fs=fs, nperseg=len(breathing_signal))
        breathing_band = (f_breath >= 0.1) & (f_breath <= 0.5) # Typical range: 6-30 breaths/min
        if np.any(breathing_band):
            peak_freq_index = np.argmax(pxx_breath[breathing_band])
            features['breathing_rate'] = f_breath[breathing_band][peak_freq_index]

    return features

def main():
    """
    Main function to extract features and create the dataset.
    """
    logging.info("Starting feature extraction for the control experiment...")

    if not os.path.exists(INPUT_BASE_DIR):
        logging.error(f"Input directory not found: {INPUT_BASE_DIR}. Please run data_generator.py first.")
        return

    all_features = []

    for category in CATEGORIES:
        logging.info(f"--- Processing category: {category} ---")

        input_cat_dir = os.path.join(INPUT_BASE_DIR, category)
        if not os.path.isdir(input_cat_dir):
            logging.warning(f"Category directory not found: {input_cat_dir}")
            continue

        files = [f for f in os.listdir(input_cat_dir) if f.endswith('.npz')]

        for i, filename in enumerate(files):
            full_npz_path = os.path.join(input_cat_dir, filename)

            try:
                data = np.load(full_npz_path)
                radar_signal = data['radar_signal']
                time_axis = data['time_axis']
                fs = 1 / (time_axis[1] - time_axis[0])

                # Use the existing signal processor
                breathing, _, rr_cleaned, pxx, f = process_gold_standard(radar_signal, fs)

                # Calculate features from the processed signals
                sample_features = calculate_features(breathing, rr_cleaned, pxx, f, fs)
                sample_features['label'] = category # Add the label
                sample_features['filename'] = filename # For traceability

                all_features.append(sample_features)

            except Exception as e:
                logging.error(f"Failed to process {filename}. Error: {e}")

    # Convert the list of feature dictionaries to a DataFrame
    feature_df = pd.DataFrame(all_features)

    # Save to CSV
    feature_df.to_csv(OUTPUT_CSV_PATH, index=False)

    logging.info(f"Feature extraction complete! Dataset saved to {OUTPUT_CSV_PATH}")
    logging.info(f"Dataset shape: {feature_df.shape}")
    logging.info(f"Sample of the dataset:\n{feature_df.head()}")

if __name__ == '__main__':
    main()
