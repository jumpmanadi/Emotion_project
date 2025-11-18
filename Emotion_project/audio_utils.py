import librosa
import numpy as np

def extract_features(y, sr):
    """
    Extracts 40 MFCC mean coefficients from an audio signal.

    Parameters:
    - y (np.array): The audio time series.
    - sr (int): The sample rate of the audio.

    Returns:
    - np.array: 1D array of length 40.
    """
    try:
        # Ensure numpy array
        y = np.array(y, dtype=float)

        # If stereo, convert to mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Basic normalization
        if np.max(np.abs(y)) > 0:
            y = y - np.mean(y)
            y = y / np.max(np.abs(y))

        # Compute MFCCs: shape (n_mfcc, time)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Take mean over time axis -> (40,)
        mfccs_mean = np.mean(mfccs, axis=1)

        return mfccs_mean

    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None
