
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import logging
import argparse
from signal_processor import process_gold_standard

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Configuration ---
INPUT_BASE_DIR = 'sedentary_dataset_py'
CATEGORIES = ['relaxed', 'flow', 'stress', 'drowsiness']
OUTPUT_DPI = 100

def create_fingerprint_image(breathing_signal, heartbeat_signal, rr_intervals, pxx, f, output_path, version='V1'):
    """
    Creates and saves a physiological fingerprint image based on the specified version. (Robust Version)
    """
    layouts = {
        'V1': {'rows': 2, 'cols': 2, 'size': (256, 256)},
        'V2': {'rows': 1, 'cols': 2, 'size': (256, 128)},
        'V3': {'rows': 1, 'cols': 2, 'size': (256, 128)},
        'V4': {'rows': 1, 'cols': 1, 'size': (128, 128)},
    }
    layout = layouts[version]
    fig, axs = plt.subplots(layout['rows'], layout['cols'],
                            figsize=(layout['size'][0]/OUTPUT_DPI, layout['size'][1]/OUTPUT_DPI),
                            squeeze=False) # squeeze=False ensures axs is always 2D
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    # Simplified and robust plotting logic
    if version == 'V1':
        if breathing_signal is not None: axs[0, 0].plot(zscore(breathing_signal))
        if heartbeat_signal is not None: axs[0, 1].plot(zscore(heartbeat_signal))
        if rr_intervals is not None: axs[1, 0].plot(rr_intervals, 'o-')
        if pxx is not None and f is not None:
            axs[1, 1].plot(f, pxx)
            axs[1, 1].set_xlim([0, 0.5])
    elif version == 'V2':
        if rr_intervals is not None: axs[0, 0].plot(rr_intervals, 'o-')
        if pxx is not None and f is not None:
            axs[0, 1].plot(f, pxx)
            axs[0, 1].set_xlim([0, 0.5])
    elif version == 'V3':
        if breathing_signal is not None: axs[0, 0].plot(zscore(breathing_signal))
        if heartbeat_signal is not None: axs[0, 1].plot(zscore(heartbeat_signal))
    elif version == 'V4':
        if pxx is not None and f is not None:
            axs[0, 0].plot(f, pxx)
            axs[0, 0].set_xlim([0, 0.5])

    # Turn off all axes
    for row in axs:
        for ax in row:
            ax.axis('off')

    plt.savefig(output_path, dpi=OUTPUT_DPI)
    plt.close(fig)

def main(version):
    """
    Main function to generate the fingerprint image dataset for a specific version.
    """
    output_base_dir = f'sedentary_images_dataset_{version.lower()}'
    logging.info(f"Starting fingerprint generation for version: {version} -> Outputting to {output_base_dir}")

    os.makedirs(output_base_dir, exist_ok=True)

    for category in CATEGORIES:
        logging.info(f"--- Processing category: {category} ---")

        input_cat_dir = os.path.join(INPUT_BASE_DIR, category)
        output_cat_dir = os.path.join(output_base_dir, category)
        os.makedirs(output_cat_dir, exist_ok=True)

        files = [f for f in os.listdir(input_cat_dir) if f.endswith('.npz')]

        for i, filename in enumerate(files):
            full_npz_path = os.path.join(input_cat_dir, filename)

            try:
                data = np.load(full_npz_path)
                radar_signal, time_axis = data['radar_signal'], data['time_axis']
                fs = 1 / (time_axis[1] - time_axis[0])

                breathing, heartbeat, rr_cleaned, pxx, f = process_gold_standard(radar_signal, fs)

                output_image_name = filename.replace('.npz', '.png')
                output_image_path = os.path.join(output_cat_dir, output_image_name)

                create_fingerprint_image(breathing, heartbeat, rr_cleaned, pxx, f, output_image_path, version)

                if (i + 1) % 50 == 0:
                    logging.info(f"Processed {i+1}/{len(files)} files in {category}")

            except Exception as e:
                logging.error(f"Failed to process {filename}. Error: {e}", exc_info=True)

    logging.info(f"--- Fingerprint dataset generation for {version} complete! ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate physiological fingerprint datasets.")
    parser.add_argument('--version', type=str, default='V1',
                        choices=['V1', 'V2', 'V3', 'V4'],
                        help="Version of the fingerprint to generate (V1, V2, V3, V4).")
    args = parser.parse_args()
    main(args.version)
