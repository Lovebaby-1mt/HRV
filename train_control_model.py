
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Configuration ---
INPUT_CSV_PATH = 'features_dataset.csv'
TEST_SIZE = 0.3
RANDOM_STATE = 42

def main():
    """
    Main function to train and evaluate the control model.
    """
    logging.info("Starting the training process for the control model...")

    # --- 2. Load and Prepare Data ---
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        logging.error(f"Dataset not found at {INPUT_CSV_PATH}. Please run feature_extractor.py first.")
        return

    # Handle potential missing values by filling with the mean of the column
    df = df.fillna(df.mean(numeric_only=True))

    # Define features (X) and target (y)
    features = [col for col in df.columns if col not in ['label', 'filename']]
    X = df[features]
    y = df['label']

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded # Ensure proportional representation of labels
    )

    logging.info(f"Data loaded and prepared. Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # --- 3. Train the Random Forest Model ---
    logging.info("Training the Random Forest Classifier...")

    # Initialize the model with balanced class weights
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )

    # Train the model
    rf_classifier.fit(X_train, y_train)

    logging.info("Model training complete.")

    # --- 4. Evaluate the Model ---
    logging.info("Evaluating the model on the test set...")

    y_pred = rf_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model Accuracy: {accuracy:.4f}")

    # Display detailed classification report
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    logging.info("Classification Report:\n" + report)

    # --- 5. Visualize Results (Confusion Matrix) ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Random Forest Control Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Save the plot
    confusion_matrix_path = 'confusion_matrix_control_model.png'
    plt.savefig(confusion_matrix_path)
    logging.info(f"Confusion matrix saved to {confusion_matrix_path}")

    # Display feature importances
    feature_importances = pd.Series(rf_classifier.feature_importances_, index=features).sort_values(ascending=False)
    logging.info("Top 5 Feature Importances:\n" + str(feature_importances.head()))


if __name__ == '__main__':
    main()
