import pandas as pd
import torch
from torch.utils.data import TensorDataset

def load_uploaded_csv_as_tensor_dataset(uploaded_file, label_column='label'):
    """
    Convert a CSV file uploaded via Streamlit into a PyTorch TensorDataset.

    Args:
        uploaded_file (UploadedFile): Streamlit-uploaded CSV file.
        label_column (str): Name of the column that contains labels.

    Returns:
        TensorDataset: dataset ready for training.
    """
    # Read CSV into DataFrame
    df = pd.read_csv(uploaded_file)

    # Separate features and label
    if label_column not in df.columns:
        raise ValueError(f"Expected label column '{label_column}' not found in dataset.")

    X = df.drop(columns=[label_column]).values.astype('float32')
    y = df[label_column].values.astype('int64')

    # Convert to tensors
    X_tensor = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y)

    return TensorDataset(X_tensor, y_tensor)
