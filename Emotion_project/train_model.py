import os
import glob
import librosa
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from audio_utils import extract_features

import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATASET_PATH = "data/ravdess"
MODEL_FILENAME = "model.pkl"
SCALER_FILENAME = "scaler.pkl"
ENCODER_FILENAME = "labelencoder.pkl"

def load_data():
    """Loads all RAVDESS data and extracts MFCC features."""

    # RAVDESS filename example: 03-01-06-01-02-01-12.wav
    # The third part (06) is the emotion.
    emotions_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    # We only keep these emotions
    observed_emotions = ['calm', 'happy', 'fearful', 'disgust', 'angry', 'sad']

    X, y = [], []

    print(f"Loading data from: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path does not exist: {DATASET_PATH}")
        return None, None

    pattern = os.path.join(DATASET_PATH, 'Actor_*', '*.wav')
    files = glob.glob(pattern)

    if not files:
        print("No .wav files found. Check your DATASET_PATH.")
        return None, None

    for file in files:
        file_name = os.path.basename(file)
        parts = file_name.split("-")
        if len(parts) < 3:
            continue

        emotion_code = parts[2]
        emotion = emotions_map.get(emotion_code)

        if emotion in observed_emotions:
            try:
                # Load audio file
                audio_data, sample_rate = librosa.load(file, res_type='kaiser_fast')
                # Extract MFCC features
                feature = extract_features(audio_data, sample_rate)

                if feature is not None:
                    X.append(feature)
                    y.append(emotion)
            except Exception as e:
                print(f"Error processing {file}: {e}")

    if not X:
        print("No audio files processed successfully.")
        return None, None

    X = np.array(X)
    y = np.array(y)

    print(f"Loaded {len(X)} audio samples with feature shape {X.shape}.")
    return X, y

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """Plot confusion matrix as a heatmap using matplotlib only."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Write numbers inside the boxes
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

def plot_feature_importance(model, feature_dim):
    """Plot feature importances for MFCC features."""
    importances = model.feature_importances_
    indices = np.arange(feature_dim)

    plt.figure(figsize=(10, 4))
    plt.bar(indices, importances)
    plt.title("RandomForest Feature Importance (MFCC Coefficients)")
    plt.xlabel("MFCC Index")
    plt.ylabel("Importance")
    plt.tight_layout()

def train_model():
    """Trains a RandomForest model and saves it along with scaler and label encoder."""

    X, y = load_data()
    if X is None or y is None:
        return

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print(f"Training RandomForestClassifier with {len(X_train)} samples...")

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # RandomForest with tuned hyperparameters
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )

    # Train
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print("\n=== EVALUATION METRICS ===")
    print(f"Accuracy on Test Set: {accuracy * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (rows = true, cols = predicted):")
    print("Label order:", label_encoder.classes_)
    print(cm)

    # PLOTS
    # Confusion matrix heatmap
    plot_confusion_matrix(cm, label_encoder.classes_, title="Confusion Matrix - Audio Emotion Model")

    # Feature importance bar plot (40 MFCCs)
    plot_feature_importance(model, feature_dim=X.shape[1])

    # Show all plots at the end
    plt.show()

    # Save for deployment
    with open(MODEL_FILENAME, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_FILENAME, 'wb') as f:
        pickle.dump(scaler, f)
    with open(ENCODER_FILENAME, 'wb') as f:
        pickle.dump(label_encoder, f)

    print(f"\nModel saved as {MODEL_FILENAME}")
    print(f"Scaler saved as {SCALER_FILENAME}")
    print(f"Label encoder saved as {ENCODER_FILENAME}")

if __name__ == "__main__":
    train_model()
