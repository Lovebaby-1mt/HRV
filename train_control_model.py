
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Configuration ---
INPUT_CSV_PATH = 'features_dataset.csv'
N_SPLITS = 10  # Number of folds for cross-validation
RANDOM_STATE = 42

def main():
    """
    Main function to train and evaluate the control model using K-Fold Cross-Validation.
    """
    logging.info("Starting a more rigorous training process for the control model using Stratified K-Fold Cross-Validation...")

    # --- 2. Load and Prepare Data ---
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        logging.error(f"Dataset not found at {INPUT_CSV_PATH}. Please run feature_extractor.py first.")
        return

    df = df.fillna(df.mean(numeric_only=True))
    features = [col for col in df.columns if col not in ['label', 'filename']]
    X = df[features]
    y = df['label']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    logging.info(f"Data loaded and prepared. Full dataset shape: {X.shape}")

    # --- 3. K-Fold Cross-Validation ---
    logging.info(f"Performing Stratified {N_SPLITS}-Fold Cross-Validation...")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    accuracies = []
    total_cm = np.zeros((len(le.classes_), len(le.classes_)), dtype=int)

    for fold, (train_index, test_index) in enumerate(skf.split(X, y_encoded)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        # Initialize and train the model for this fold
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced')
        rf_classifier.fit(X_train, y_train)

        # Evaluate
        y_pred = rf_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        # Aggregate confusion matrix
        total_cm += confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_)))

        logging.info(f"Fold {fold+1}/{N_SPLITS} - Accuracy: {accuracy:.4f}")

    # --- 4. Report Aggregated Results ---
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    logging.info("-" * 40)
    logging.info("Cross-Validation Summary:")
    logging.info(f"Mean Accuracy: {mean_accuracy:.4f}")
    logging.info(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")
    logging.info("-" * 40)

    # --- 5. Visualize Aggregated Confusion Matrix ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Aggregated Confusion Matrix ({N_SPLITS}-Fold Cross-Validation)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    confusion_matrix_path = 'confusion_matrix_control_model_cv.png'
    plt.savefig(confusion_matrix_path)
    logging.info(f"Aggregated confusion matrix saved to {confusion_matrix_path}")

    # --- 6. Save Results for Comparative Analysis ---
    results = {
        'accuracies': accuracies,
        'confusion_matrix': total_cm.tolist(),
        'labels': le.classes_.tolist()
    }
    with open('rf_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    logging.info("Random Forest results saved to rf_results.json")

    # --- 7. Final Model and Feature Importances ---
    logging.info("Training a final model on the entire dataset to determine feature importances...")
    final_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced')
    final_model.fit(X, y_encoded)

    feature_importances = pd.Series(final_model.feature_importances_, index=features).sort_values(ascending=False)
    logging.info("Top 5 Feature Importances (from model trained on full dataset):\n" + str(feature_importances.head()))


if __name__ == '__main__':
    main()
