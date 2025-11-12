
import os
import numpy as np
from scipy.signal import windows
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Configuration ---
OUTPUT_DIR = 'sedentary_dataset_py'
STATES = ['relaxed', 'flow', 'stress', 'drowsiness']
SIMULATION_DURATION = 180  # seconds
SAMPLES_PER_STATE = 30 # Reduced for faster generation, can be increased to 300
RADAR_FS = 100  # Hz
HR_RANGE = [60, 80]  # bpm

def simulate_rr_intervals_sedentary(duration, state, mean_hr):
    """
    Simulates heart beat timings based on physiological state.
    Translates simulate_rr_intervals_sedentary from MATLAB.
    """
    if state == 'relaxed':
        rmssd_ms = 70 + np.random.randn() * 10
        lf_hf_ratio = 0.6 + np.random.randn() * 0.2
    elif state == 'flow':
        rmssd_ms = 45 + np.random.randn() * 8
        lf_hf_ratio = 2.0 + np.random.randn() * 0.4
    elif state == 'stress':
        rmssd_ms = 20 + np.random.randn() * 5
        lf_hf_ratio = 4.5 + np.random.randn() * 0.5
    else:  # drowsiness
        rmssd_ms = 30 + np.random.randn() * 5
        lf_hf_ratio = 3.5 + np.random.randn() * 0.5

    internal_fs = 100
    time_mod = np.arange(0, duration, 1/internal_fs)

    hf_amplitude = rmssd_ms / np.sqrt(2)
    hf_freq = 0.25
    hf_component = hf_amplitude * np.sin(2 * np.pi * hf_freq * time_mod)

    # Ensure lf_hf_ratio is non-negative before taking the square root
    if lf_hf_ratio < 0:
        lf_hf_ratio = 0
    lf_amplitude = hf_amplitude * np.sqrt(lf_hf_ratio)
    lf_freq = 0.1
    lf_component = lf_amplitude * np.sin(2 * np.pi * lf_freq * time_mod)

    mean_rr = 60000 / mean_hr
    noise = 10 * np.random.randn(len(time_mod))

    rr_modulation = mean_rr + hf_component + lf_component + noise

    heartbeats_time = [0.0]
    while heartbeats_time[-1] < duration:
        current_time = heartbeats_time[-1]
        current_rr_idx = int(round(current_time * internal_fs))

        if current_rr_idx >= len(rr_modulation):
            break

        current_rr = rr_modulation[current_rr_idx]
        next_beat = current_time + max(current_rr, 200) / 1000

        if next_beat >= duration:
            break

        heartbeats_time.append(next_beat)

    return np.array(heartbeats_time)

def generate_chest_displacement_sedentary(heartbeats_time, duration, fs, state):
    """
    Generates the simulated radar signal based on chest displacement.
    Translates generate_chest_displacement_sedentary from MATLAB.
    """
    n_samples = int(duration * fs)
    time_axis = np.linspace(0, duration, n_samples)

    if state in ['relaxed', 'drowsiness']:
        breathing_freq = 0.2
        breathing_amplitude = 1.2
    elif state == 'stress':
        breathing_freq = 0.33
        breathing_amplitude = 0.7
    else:  # flow
        breathing_freq = 0.25
        breathing_amplitude = 1.0

    breathing_signal = breathing_amplitude * np.sin(2 * np.pi * breathing_freq * time_axis)

    heartbeat_signal = np.zeros(n_samples)
    heartbeat_amplitude = 0.2

    for t in heartbeats_time:
        idx = int(round(t * fs))
        if 0 <= idx < n_samples:
            pulse_width = int(round(0.1 * fs))
            # Use scipy's gaussian window, std dev is width/4
            window = windows.gaussian(pulse_width, std=pulse_width/4)

            start_idx = max(0, idx - pulse_width // 2)
            end_idx = min(n_samples, start_idx + pulse_width)

            win_len = end_idx - start_idx
            heartbeat_signal[start_idx:end_idx] += heartbeat_amplitude * window[:win_len]

    combined_signal = breathing_signal + heartbeat_signal
    baseline_wander = 0.1 * np.sin(2 * np.pi * 0.05 * time_axis)
    white_noise = 0.05 * np.random.randn(n_samples)

    final_signal = combined_signal + baseline_wander + white_noise
    return time_axis, final_signal

def main():
    """
    Main function to generate the dataset.
    """
    logging.info("Starting dataset generation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_files_generated = 0

    for state in STATES:
        state_dir = os.path.join(OUTPUT_DIR, state)
        os.makedirs(state_dir, exist_ok=True)
        logging.info(f"--- Generating state: {state} ---")

        for sample_num in range(1, SAMPLES_PER_STATE + 1):
            mean_hr = np.random.randint(HR_RANGE[0], HR_RANGE[1] + 1)

            heartbeats = simulate_rr_intervals_sedentary(SIMULATION_DURATION, state, mean_hr)
            time_axis, radar_signal = generate_chest_displacement_sedentary(heartbeats, SIMULATION_DURATION, RADAR_FS, state)

            label = state
            rr_intervals_ms = np.diff(heartbeats) * 1000

            filename = f'{state}_hr{mean_hr}_sample{sample_num:03d}.npz'
            filepath = os.path.join(state_dir, filename)

            np.savez_compressed(
                filepath,
                radar_signal=radar_signal,
                time_axis=time_axis,
                label=label,
                rr_intervals_ms=rr_intervals_ms,
                heartbeats=heartbeats
            )

            total_files_generated += 1
            if sample_num % 10 == 0:
                logging.info(f"Generated {sample_num}/{SAMPLES_PER_STATE} samples for {state}")

    logging.info(f"Dataset generation complete! Total files generated: {total_files_generated}")

if __name__ == '__main__':
    main()
