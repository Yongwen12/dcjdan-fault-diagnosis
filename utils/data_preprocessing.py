import numpy as np
import pandas as pd
from scipy import signal

def segment_signal(data, segment_length=2048):
    """Segment data into fixed length segments"""
    segments = []
    for i in range(0, len(data) - segment_length + 1, segment_length):
        segments.append(data[i:i+segment_length])
    return np.array(segments)

def apply_filter(data, low_cutoff=0.1, high_cutoff=30, fs=1000):
    """Apply bandpass filter"""
    sos = signal.butter(4, [low_cutoff, high_cutoff], btype='band', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, data)
    return filtered

def preprocess_dataset(raw_data, label, segment_length=512):
    """
    Full preprocessing pipeline:
    1. Filter raw signal
    2. Segment into fixed length
    3. Generate labeled DataFrame
    """
    filtered = apply_filter(raw_data)
    segmented = segment_signal(filtered, segment_length)

    # Repeat labels for each segment
    labels = np.full((segmented.shape[0], 1), label)

    # Combine features and label
    dataset = np.hstack((segmented, labels))

    # Create DataFrame with feature columns and label column
    feature_cols = [f"feature_{i}" for i in range(segment_length)]
    df = pd.DataFrame(dataset, columns=feature_cols + ["label"])

    return df

# ==== Example usage ====
if __name__ == "__main__":
    # Fake raw data for 3 classes
    all_dfs = []
    for lbl in range(3):
        raw_signal = np.random.randn(10000)  # Simulated raw signal
        df = preprocess_dataset(raw_signal, label=lbl, segment_length=512)
        all_dfs.append(df)

    # Merge all data into one CSV
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv("processed_dataset.csv", index=False)
    print("Saved processed dataset to processed_dataset.csv")
