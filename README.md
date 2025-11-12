# Robust Physiological State Sensing via "Physiological Fingerprints" and CNN

This project develops a novel, robust algorithm for classifying four subtle sedentary physiological states (relaxed, flow, stress, drowsiness) from non-contact millimeter-wave radar signals. It moves beyond fragile, hand-crafted feature engineering by introducing the concept of a "Physiological Fingerprint," a 2D image representation of physiological signals, which is then classified by a Convolutional Neural Network (CNN).

## Core Concept: The "Physiological Fingerprint"

Traditional methods relying on numerical HRV (Heart Rate Variability) features like the LF/HF ratio are extremely sensitive to minor inaccuracies in peak detection from the raw signal. A small error can lead to a completely wrong physiological conclusion.

Our approach, inspired by multi-spectral analysis, pivots from *calculating fragile numbers* to *learning robust patterns*. We represent a 30-second physiological snapshot as a 2x2 image, the "Physiological Fingerprint."

This image consists of four plots:
- **(Top-Left)** Breathing Waveform
- **(Top-Right)** Heartbeat Waveform
- **(Bottom-Left)** RR-Interval Tachogram
- **(Bottom-Right)** HRV Power Spectral Density (PSD)

Our central hypothesis is that a CNN can learn the global, robust *morphological patterns* in these images that correspond to each physiological state, effectively ignoring the minor "glitches" (like outlier points in the tachogram) that would invalidate traditional numerical methods.

## Repository Structure

This project has been fully implemented in Python to create an end-to-end pipeline from data simulation to model comparison.

-   `data_generator.py`: Simulates and generates high-fidelity `.npz` data files for the four physiological states.
-   `signal_processor.py`: Contains the core signal processing logic to extract intermediate waveforms (breathing, heartbeat, RR intervals, PSD) from the raw data.
-   `fingerprint_generator.py`: Uses the above scripts to convert the simulated `.npz` data into the 2D "Physiological Fingerprint" `.png` images.
-   `feature_extractor.py`: A parallel pipeline that extracts traditional numerical features (e.g., SDNN, RMSSD, LF/HF ratio) from the data to create a `.csv` dataset for our baseline model.
-   `train_control_model.py`: Trains and evaluates a strong baseline model (Random Forest) on the numerical features using Stratified K-Fold cross-validation.
-   `train_cnn.py`: Trains and evaluates our primary CNN model on the "Physiological Fingerprint" images, also using Stratified K-Fold cross-validation for a fair comparison.
-   `generate_comparative_plots.py`: A dedicated script to load the saved results from both models and generate a suite of publication-quality comparative visualizations.
-   `requirements.txt`: Contains all necessary Python packages to reproduce the environment.

## How to Run the Complete Workflow

To replicate the entire experiment from start to finish, follow these steps in order.

**Step 1: Set up the environment**
```bash
pip install -r requirements.txt
```

**Step 2: Generate the simulated raw data and fingerprint images**
```bash
python3 data_generator.py
python3 fingerprint_generator.py
```

**Step 3: Extract features and train the baseline model**
```bash
python3 feature_extractor.py
python3 train_control_model.py
```
This will train the Random Forest, evaluate it, and save its results to `rf_results.json`.

**Step 4: Train the CNN model**
```bash
python3 train_cnn.py
```
This will train the CNN, evaluate it, and save its results to `cnn_results.json`.
*Note: The current `train_cnn.py` contains an advanced architecture that is computationally intensive and may time out in resource-constrained environments.*

**Step 5: Generate comparative visualizations**
```bash
python3 generate_comparative_plots.py
```
This will load the `.json` results from the two models and generate the final comparison plots (`performance_distribution.png`, `normalized_confusion_matrices.png`, etc.).

## Current Results & Future Work

Our current experiments show:
-   **Baseline (Random Forest):** A very strong performance with a mean accuracy of **~89.7%** via 10-fold cross-validation.
-   **Initial CNN:** A proof-of-concept CNN achieved a mean accuracy of **~73.3%**.

This performance gap highlights a key area for future work. The `train_cnn.py` script already contains a more advanced, deeper architecture with data augmentation and batch normalization. The primary next step is to train this model in a more powerful computational environment (e.g., a cloud GPU instance) to unlock the full potential of the "Physiological Fingerprint" approach and likely surpass the strong baseline.
