
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Configuration ---
IMAGE_DIR = 'sedentary_images_dataset_py'
CATEGORIES = ['relaxed', 'flow', 'stress', 'drowsiness']
IMG_SIZE = (128, 128)  # Smaller size for faster training
N_SPLITS = 10
BATCH_SIZE = 32
EPOCHS = 15
RANDOM_STATE = 42

def create_cnn_model(num_classes):
    """Defines and compiles a simple CNN model."""
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Rescaling(1./255),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_and_preprocess_image(path, label):
    """Loads and preprocesses a single image."""
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return image, label

def create_dataset(paths, labels):
    """Creates a tf.data.Dataset from file paths and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    image_label_ds = tf.data.Dataset.zip((path_ds, label_ds))
    return image_label_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

def main():
    """Main function to train and evaluate the CNN model."""
    logging.info("Starting CNN training with Stratified K-Fold Cross-Validation.")

    # --- 2. Load File Paths and Labels ---
    if not os.path.exists(IMAGE_DIR):
        logging.error(f"Image directory not found: {IMAGE_DIR}. Please run fingerprint_generator.py first.")
        return

    all_filepaths = []
    all_labels = []
    for category in CATEGORIES:
        cat_dir = os.path.join(IMAGE_DIR, category)
        for fname in os.listdir(cat_dir):
            if fname.endswith('.png'):
                all_filepaths.append(os.path.join(cat_dir, fname))
                all_labels.append(category)

    all_filepaths = np.array(all_filepaths)
    all_labels = np.array(all_labels)

    le = LabelEncoder()
    y_encoded = le.fit_transform(all_labels)
    num_classes = len(le.classes_)

    # --- 3. K-Fold Cross-Validation ---
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    accuracies = []
    total_cm = np.zeros((num_classes, num_classes), dtype=int)

    for fold, (train_index, test_index) in enumerate(skf.split(all_filepaths, y_encoded)):
        logging.info(f"--- Starting Fold {fold+1}/{N_SPLITS} ---")

        # Split data for this fold
        X_train_paths, X_test_paths = all_filepaths[train_index], all_filepaths[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        # Create tf.data datasets
        train_ds = create_dataset(X_train_paths, y_train).shuffle(buffer_size=len(y_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        test_ds = create_dataset(X_test_paths, y_test).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Create and train the model
        model = create_cnn_model(num_classes)
        model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds, verbose=0)

        # Evaluate
        loss, accuracy = model.evaluate(test_ds, verbose=0)
        accuracies.append(accuracy)

        # Aggregate confusion matrix
        y_pred_probs = model.predict(test_ds, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        total_cm += confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))

        logging.info(f"Fold {fold+1}/{N_SPLITS} - Validation Accuracy: {accuracy:.4f}")

    # --- 4. Report Aggregated Results ---
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    logging.info("-" * 40)
    logging.info("CNN Cross-Validation Summary:")
    logging.info(f"Mean Accuracy: {mean_accuracy:.4f}")
    logging.info(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")
    logging.info("-" * 40)

    # --- 5. Visualize and Save Results ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Aggregated CNN Confusion Matrix ({N_SPLITS}-Fold CV)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix_cnn_model_cv.png')
    logging.info("Aggregated CNN confusion matrix saved to confusion_matrix_cnn_model_cv.png")

    # --- 6. Save Results for Comparative Analysis ---
    results = {
        'accuracies': accuracies,
        'confusion_matrix': total_cm.tolist(),
        'labels': le.classes_.tolist()
    }
    with open('cnn_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    logging.info("CNN results saved to cnn_results.json")


if __name__ == '__main__':
    # Suppress TensorFlow informational messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    main()
