
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_performance_distribution(rf_data, cnn_data, output_path='performance_distribution.png'):
    """Generates a box plot comparing the accuracy distributions of the two models."""
    logging.info("Generating performance distribution box plot...")

    data_to_plot = [rf_data['accuracies'], cnn_data['accuracies']]

    plt.figure(figsize=(10, 7))
    sns.boxplot(data=data_to_plot, palette=['skyblue', 'lightgreen'])
    plt.xticks([0, 1], ['Random Forest', 'CNN'])
    plt.ylabel('Accuracy')
    plt.title('Model Performance Distribution (10-Fold Cross-Validation)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_path)
    logging.info(f"Performance distribution plot saved to {output_path}")
    plt.close()

def plot_normalized_confusion_matrices(rf_data, cnn_data, output_path='normalized_confusion_matrices.png'):
    """Generates a side-by-side comparison of normalized confusion matrices."""
    logging.info("Generating normalized confusion matrix comparison plot...")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Random Forest Matrix
    cm_rf = np.array(rf_data['confusion_matrix'])
    cm_rf_normalized = cm_rf.astype('float') / cm_rf.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_rf_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[0],
                xticklabels=rf_data['labels'], yticklabels=rf_data['labels'])
    axes[0].set_title('Random Forest (Normalized)')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # CNN Matrix
    cm_cnn = np.array(cnn_data['confusion_matrix'])
    cm_cnn_normalized = cm_cnn.astype('float') / cm_cnn.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_cnn_normalized, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
                xticklabels=cnn_data['labels'], yticklabels=cnn_data['labels'])
    axes[1].set_title('CNN (Normalized)')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    fig.suptitle('Normalized Confusion Matrix Comparison (Recall %)', fontsize=16)
    plt.savefig(output_path)
    logging.info(f"Normalized confusion matrices plot saved to {output_path}")
    plt.close()

def plot_per_class_metrics(rf_data, cnn_data, output_path='per_class_metrics.png'):
    """Generates a grouped bar chart comparing the F1-scores of the two models for each class."""
    logging.info("Generating per-class F1-score comparison plot...")

    # Calculate metrics for Random Forest
    cm_rf = np.array(rf_data['confusion_matrix'])
    precision_rf = np.diag(cm_rf) / np.sum(cm_rf, axis=0)
    recall_rf = np.diag(cm_rf) / np.sum(cm_rf, axis=1)
    f1_rf = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)

    # Calculate metrics for CNN
    cm_cnn = np.array(cnn_data['confusion_matrix'])
    precision_cnn = np.diag(cm_cnn) / np.sum(cm_cnn, axis=0)
    recall_cnn = np.diag(cm_cnn) / np.sum(cm_cnn, axis=1)
    f1_cnn = 2 * (precision_cnn * recall_cnn) / (precision_cnn + recall_cnn)

    # Create DataFrame for plotting
    labels = rf_data['labels']
    df = pd.DataFrame({
        'Class': labels,
        'Random Forest': f1_rf,
        'CNN': f1_cnn
    })

    df_melted = df.melt(id_vars='Class', var_name='Model', value_name='F1-Score')

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Class', y='F1-Score', hue='Model', data=df_melted, palette=['skyblue', 'lightgreen'])
    plt.title('Per-Class F1-Score Comparison')
    plt.ylabel('F1-Score')
    plt.xlabel('Physiological State')
    plt.ylim(0, 1)
    plt.legend(title='Model')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_path)
    logging.info(f"Per-class metrics plot saved to {output_path}")
    plt.close()


def main():
    """Main function to generate all comparative plots."""
    try:
        with open('rf_results.json', 'r') as f:
            rf_data = json.load(f)
        with open('cnn_results.json', 'r') as f:
            cnn_data = json.load(f)
    except FileNotFoundError as e:
        logging.error(f"Result file not found: {e}. Please run the training scripts first.")
        return

    plot_performance_distribution(rf_data, cnn_data)
    plot_normalized_confusion_matrices(rf_data, cnn_data)
    plot_per_class_metrics(rf_data, cnn_data)

    logging.info("All comparative plots have been generated successfully.")

if __name__ == '__main__':
    main()
