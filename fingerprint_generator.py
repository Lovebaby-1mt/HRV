
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import logging
from signal_processor import process_gold_standard

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Configuration ---
INPUT_BASE_DIR = 'sedentary_dataset_py'
OUTPUT_BASE_DIR = 'sedentary_images_dataset_py'
CATEGORIES = ['relaxed', 'flow', 'stress', 'drowsiness']
OUTPUT_SIZE_PX = (256, 256)
OUTPUT_DPI = 100 # Adjusted for better control over pixel size

def create_fingerprint_image(breathing_signal, heartbeat_signal, rr_intervals, pxx, f, output_path):
    """
    Creates and saves a 2x2 physiological fingerprint image.
    """
    try:
        fig, axs = plt.subplots(2, 2, figsize=(OUTPUT_SIZE_PX[0]/OUTPUT_DPI, OUTPUT_SIZE_PX[1]/OUTPUT_DPI))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        # (1,1) Breathing Signal
        if breathing_signal is not None and len(breathing_signal) > 0:
            axs[0, 0].plot(zscore(breathing_signal))
        axs[0, 0].axis('off')

        # (1,2) Heartbeat Signal
        if heartbeat_signal is not None and len(heartbeat_signal) > 0:
            axs[0, 1].plot(zscore(heartbeat_signal))
        axs[0, 1].axis('off')

        # (2,1) RR Intervals
        if rr_intervals is not None and len(rr_intervals) > 0:
            axs[1, 0].plot(rr_intervals, 'o-')
        axs[1, 0].axis('off')

        # (2,2) Power Spectral Density (PSD)
        if pxx is not None and f is not None and len(pxx) > 0:
            axs[1, 1].plot(f, pxx)
            axs[1, 1].set_xlim([0, 0.5]) # Match MATLAB version
        axs[1, 1].axis('off')

        plt.savefig(output_path, dpi=OUTPUT_DPI)
        plt.close(fig)

    except Exception as e:
        logging.error(f"Failed to create fingerprint image for {output_path}. Error: {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)


def main():
    """
    Main function to generate the fingerprint image dataset.
    """
    logging.info("Starting physiological fingerprint dataset generation...")

    if not os.path.exists(INPUT_BASE_DIR):
        logging.error(f"Input directory not found: {INPUT_BASE_DIR}. Please run data_generator.py first.")
        return

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    for category in CATEGORIES:
        logging.info(f"--- Processing category: {category} ---")

        input_cat_dir = os.path.join(INPUT_BASE_DIR, category)
        output_cat_dir = os.path.join(OUTPUT_BASE_DIR, category)
        os.makedirs(output_cat_dir, exist_ok=True)

        if not os.path.isdir(input_cat_dir):
            logging.warning(f"Category directory not found: {input_cat_dir}")
            continue

        files = [f for f in os.listdir(input_cat_dir) if f.endswith('.npz')]

        for i, filename in enumerate(files):
            full_npz_path = os.path.join(input_cat_dir, filename)

            try:
                # Load data
                data = np.load(full_npz_path)
                radar_signal = data['radar_signal']
                time_axis = data['time_axis']
                fs = 1 / (time_axis[1] - time_axis[0])

                # Process signal to get intermediate representations
                breathing, heartbeat, rr_cleaned, pxx, f = process_gold_standard(radar_signal, fs)

                # Define output path
                output_image_name = filename.replace('.npz', '.png')
                output_image_path = os.path.join(output_cat_dir, output_image_name)

                # Create and save the image
                create_fingerprint_image(breathing, heartbeat, rr_cleaned, pxx, f, output_image_path)

                if (i + 1) % 10 == 0:
                    logging.info(f"Processed {i+1}/{len(files)} files in {category}")

            except Exception as e:
                logging.error(f"Failed to process {filename}. Error: {e}")

    logging.info("--- Physiological fingerprint dataset generation complete! ---")

if __name__ == '__main__':
    main()
