
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import argparse
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_results(files):
    """Loads multiple JSON result files and compiles them."""
    results = {}
    for file in files:
        try:
            version = os.path.basename(file).split('_')[2].split('.')[0].upper()
            with open(file, 'r') as f:
                results[version] = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load or parse {file}: {e}")
    return results

def plot_accuracy_comparison(results_dict, output_path='version_accuracy_comparison.png'):
    """Generates a combined box and bar plot for accuracy comparison."""
    logging.info("Generating accuracy comparison plot...")

    versions = sorted(results_dict.keys())
    mean_accuracies = [np.mean(results_dict[v]['accuracies']) for v in versions]
    all_accuracies = [results_dict[v]['accuracies'] for v in versions]

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Bar plot for mean accuracies
    sns.barplot(x=versions, y=mean_accuracies, ax=ax1, alpha=0.6, palette='viridis')
    ax1.set_ylabel('Mean Accuracy')
    ax1.set_xlabel('Fingerprint Version')
    ax1.set_ylim(0, 1)

    # Box plot for distribution, overlaid
    ax2 = ax1.twinx()
    sns.boxplot(data=all_accuracies, ax=ax2, palette='pastel')
    ax2.set_yticklabels([]) # Hide y-axis labels
    ax2.set_yticks([])

    plt.title('Mean Accuracy and Distribution Across Fingerprint Versions')
    plt.savefig(output_path)
    logging.info(f"Accuracy comparison plot saved to {output_path}")
    plt.close()

def print_summary_table(results_dict):
    """Prints a summary table of the results to the console."""
    logging.info("Generating results summary table...")

    summary = []
    for version, data in sorted(results_dict.items()):
        mean_acc = np.mean(data['accuracies'])
        std_acc = np.std(data['accuracies'])
        summary.append({'Version': version, 'Mean Accuracy': f"{mean_acc:.4f}", 'Std Dev': f"{std_acc:.4f}"})

    df = pd.DataFrame(summary)
    logging.info("--- Experiment Summary ---")
    print(df.to_string(index=False))
    logging.info("--------------------------")


def main(result_files):
    """Main function to generate all comparative plots."""
    if not result_files:
        logging.error("No result files found. Please run the training scripts first.")
        return

    results = load_results(result_files)

    if not results:
        logging.error("Failed to load any valid result files.")
        return

    plot_accuracy_comparison(results)
    print_summary_table(results)

    logging.info("All comparative plots have been generated successfully.")

if __name__ == '__main__':
    # Use glob to find all cnn_results_v*.json files
    files = glob.glob('cnn_results_v*.json')
    main(files)
