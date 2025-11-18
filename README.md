# Robust Physiological State Sensing via "Physiological Fingerprints" and CNN

This project develops a novel, robust algorithm for classifying four subtle sedentary physiological states (relaxed, flow, stress, drowsiness) from non-contact millimeter-wave radar signals. It moves beyond fragile, hand-crafted feature engineering by introducing the concept of a "Physiological Fingerprint," a 2D image representation of physiological signals, which is then classified by a Convolutional Neural Network (CNN).

## Core Concept: The "Physiological Fingerprint"

Traditional methods relying on numerical HRV (Heart Rate Variability) features are extremely sensitive to minor inaccuracies in peak detection. Our approach pivots from *calculating fragile numbers* to *learning robust patterns*. We represent a physiological snapshot as a 2D image, the "Physiological Fingerprint," and train a CNN to learn the global, morphological patterns associated with each state.

### Fingerprint Versions for Ablation Study

To identify the most effective combination of physiological information, we experiment with four different versions of the fingerprint:

-   **V1 (Full Info):** A 2x2 grid containing Breathing Waveform, Heartbeat Waveform, RR-Interval Tachogram, and HRV Power Spectral Density (PSD).
-   **V2 (HRV-Only):** A 1x2 grid containing only the RR-Interval Tachogram and HRV PSD.
-   **V3 (Waveform-Only):** A 1x2 grid containing only the Breathing and Heartbeat Waveforms.
-   **V4 (PSD-Only):** A 1x1 grid containing only the HRV PSD, the most condensed representation.

## Repository Structure

This project is an end-to-end Python pipeline for data simulation, model training, and comparative analysis.

-   `data_generator.py`: Simulates and generates high-fidelity `.npz` data files for the physiological states.
-   `signal_processor.py`: Contains the core logic to extract intermediate waveforms (breathing, heartbeat, RR intervals, PSD).
-   `fingerprint_generator.py`: A parameterized script (use `--version`) to generate the different versions (V1-V4) of fingerprint images.
-   `feature_extractor.py`: Extracts traditional numerical features for the baseline model.
-   `train_control_model.py`: Trains and evaluates a strong baseline model (Random Forest) using Stratified K-Fold cross-validation.
-   `train_cnn.py`: A parameterized script (use `--dataset_dir` and `--output_file`) to train and evaluate the CNN on any version of the fingerprint dataset.
-   `compare_versions.py`: A dedicated script to automatically load all versioned CNN results and generate final comparative visualizations.
-   `requirements.txt`: A flexible list of Python packages to reproduce the environment.

## How to Run the Complete Workflow

This workflow allows for a full comparative analysis of the different fingerprint versions.

**Step 1: Set up the environment**
```bash
pip install -r requirements.txt
```

**Step 2: Generate the base raw data**
```bash
python data_generator.py
```

**Step 3: Generate all versions of the fingerprint image datasets**
```bash
# Generate each version sequentially
python fingerprint_generator.py --version V1
python fingerprint_generator.py --version V2
python fingerprint_generator.py --version V3
python fingerprint_generator.py --version V4
```

**Step 4: Train the CNN model for each fingerprint version**
This is a computationally intensive step. It is **highly recommended to run this in a GPU environment** (e.g., Google Colab).
```bash
# Train on each dataset sequentially
python train_cnn.py --dataset_dir sedentary_images_dataset_v1 --output_file cnn_results_v1.json
python train_cnn.py --dataset_dir sedentary_images_dataset_v2 --output_file cnn_results_v2.json
python train_cnn.py --dataset_dir sedentary_images_dataset_v3 --output_file cnn_results_v3.json
python train_cnn.py --dataset_dir sedentary_images_dataset_v4 --output_file cnn_results_v4.json
```

**Step 5: Generate the final comparative visualizations**
This script will automatically find all `cnn_results_v*.json` files and compare them.
```bash
python compare_versions.py
```
This will print a summary table to the console and save a visual comparison chart to `version_accuracy_comparison.png`.
