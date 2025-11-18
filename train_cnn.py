
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import argparse
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Configuration ---
CATEGORIES = ['relaxed', 'flow', 'stress', 'drowsiness']
N_SPLITS = 10
BATCH_SIZE = 32
EPOCHS = 20
RANDOM_STATE = 42

def get_image_size(dataset_dir):
    """Dynamically determines the image size from the first image in the dataset."""
    for category in CATEGORIES:
        cat_dir = os.path.join(dataset_dir, category)
        if os.path.exists(cat_dir) and len(os.listdir(cat_dir)) > 0:
            first_image_path = os.path.join(cat_dir, os.listdir(cat_dir)[0])
            with Image.open(first_image_path) as img:
                return img.size # (width, height)
    return (128, 128) # Fallback

def create_cnn_model(num_classes, img_height, img_width):
    """Defines and compiles an optimized CNN model, now with dynamic input size."""
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ])

    model = models.Sequential([
        layers.Input(shape=(img_height, img_width, 3)),
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(32, (3, 3), padding='same'), layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), padding='same'), layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), padding='same'), layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPooling2D(),
        layers.Conv2D(256, (3, 3), padding='same'), layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_and_preprocess_image(path, label, img_size):
    """Loads and preprocesses a single image with a dynamic size."""
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, img_size)
    return image, label

def create_dataset(paths, labels, img_size):
    """Creates a tf.data.Dataset with a dynamic image size."""
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    image_label_ds = tf.data.Dataset.zip((path_ds, label_ds))
    # Use a lambda to pass the img_size argument to the map function
    return image_label_ds.map(lambda path, label: load_and_preprocess_image(path, label, img_size),
                            num_parallel_calls=tf.data.AUTOTUNE)

def main(dataset_dir, output_file):
    """Main function to train and evaluate the CNN model on a specified dataset."""
    logging.info(f"Starting CNN training for dataset: {dataset_dir}")

    # Dynamically get image size
    img_width, img_height = get_image_size(dataset_dir)
    img_size = (img_height, img_width)
    logging.info(f"Detected image size: {img_size}")

    all_filepaths, all_labels = [], []
    for category in CATEGORIES:
        cat_dir = os.path.join(dataset_dir, category)
        for fname in os.listdir(cat_dir):
            if fname.endswith('.png'):
                all_filepaths.append(os.path.join(cat_dir, fname))
                all_labels.append(category)

    all_filepaths, all_labels = np.array(all_filepaths), np.array(all_labels)
    le = LabelEncoder()
    y_encoded = le.fit_transform(all_labels)
    num_classes = len(le.classes_)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accuracies, total_cm = [], np.zeros((num_classes, num_classes), dtype=int)

    for fold, (train_index, test_index) in enumerate(skf.split(all_filepaths, y_encoded)):
        logging.info(f"--- Starting Fold {fold+1}/{N_SPLITS} ---")

        X_train_paths, X_test_paths = all_filepaths[train_index], all_filepaths[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        train_ds = create_dataset(X_train_paths, y_train, img_size).shuffle(len(y_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        test_ds = create_dataset(X_test_paths, y_test, img_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        model = create_cnn_model(num_classes, img_height, img_width)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=0)

        model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds, callbacks=[lr_scheduler], verbose=0)

        loss, accuracy = model.evaluate(test_ds, verbose=0)
        accuracies.append(accuracy)

        y_pred_probs = model.predict(test_ds, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        total_cm += confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))

        logging.info(f"Fold {fold+1}/{N_SPLITS} - Validation Accuracy: {accuracy:.4f}")

    mean_accuracy, std_accuracy = np.mean(accuracies), np.std(accuracies)
    logging.info(f"Final Mean Accuracy: {mean_accuracy:.4f} +/- {std_accuracy:.4f}")

    results = {'accuracies': accuracies, 'confusion_matrix': total_cm.tolist(), 'labels': le.classes_.tolist()}
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"CNN results saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a CNN on a specified fingerprint dataset.")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Directory of the image dataset to train on.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the JSON results file.")
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Suppress TensorFlow info messages
    main(args.dataset_dir, args.output_file)
