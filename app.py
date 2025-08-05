import streamlit as st
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from train import train_dcjdan
from utils import load_uploaded_csv_as_tensor_dataset  # You need to implement this

st.title("Interactive DCJDAN Training Panel")

# =============================
# Dataset selection
# =============================
source_file = st.file_uploader("Upload Source Dataset (CSV)", type=["csv"])
target_file = st.file_uploader("Upload Target Dataset (CSV)", type=["csv"])

# =============================
# Model selection
# =============================
model_name = st.selectbox("Select Model", ["DCJDAN"])

# =============================
# Hyperparameters
# =============================
use_defaults = st.checkbox("Use Default Training Parameters", value=True)

if use_defaults:
    epochs = 10
    learning_rate = 0.001
    batch_size = 64
    st.info(f"Using defaults: Epochs={epochs}, LR={learning_rate}, Batch={batch_size}")
else:
    st.markdown("👉 Manually set hyperparameters")
    epochs = st.slider("Epochs", min_value=1, max_value=100, value=10)
    learning_rate = st.number_input("Learning Rate", value=0.001, format="%.5f")
    batch_size = st.slider("Batch Size", min_value=8, max_value=256, value=64)

# =============================
# Training button
# =============================
if st.button("Start Training"):
    with st.spinner("Training in progress..."):
        # Load uploaded datasets or fallback to default
        source_data = load_uploaded_csv_as_tensor_dataset(source_file) if source_file else None
        target_data = load_uploaded_csv_as_tensor_dataset(target_file) if target_file else None

        # Train model
        loss_values, features_per_epoch = train_dcjdan(
            epochs=epochs,
            lr=learning_rate,
            batch_size=batch_size,
            source_data=source_data,
            target_data=target_data
        )

    # =============================
    # Loss visualization
    # =============================
    plt.figure()
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    st.pyplot(plt)

    # =============================
    # t-SNE visualization
    # =============================
    for epoch, (f_s, f_t) in enumerate(features_per_epoch):
        if epoch % 5 == 0:
            labels = torch.cat([
                torch.zeros(len(f_s)),   # source
                torch.ones(len(f_t))     # target
            ])
            combined_features = torch.cat([f_s, f_t])
            reduced = TSNE(n_components=2).fit_transform(combined_features)
            plt.figure()
            plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='coolwarm', s=10)
            plt.title(f"t-SNE at Epoch {epoch+1}")
            st.pyplot(plt)

    st.success("Training completed successfully!")
