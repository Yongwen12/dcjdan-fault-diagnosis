import streamlit as st
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model.dcjdan_model import DCJDAN
from utils import load_uploaded_csv_as_tensor_dataset, create_dummy_dataset, mmd_loss, l2_reg_loss
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

st.title("Interactive Model Training Panel")

# Dataset upload or default option
source_file = st.file_uploader("Upload Source Dataset (CSV)", type=["csv"])
target_file = st.file_uploader("Upload Target Dataset (CSV)", type=["csv"])

# Model selection
model_name = st.selectbox("Select Model", ["DCJDAN"])

# Hyperparameters
use_defaults = st.checkbox("Use Default Training Parameters", value=True)

if use_defaults:
    epochs = 10
    learning_rate = 0.001
    batch_size = 64
else:
    epochs = st.slider("Epochs", 1, 100, 10)
    learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001)
    batch_size = st.slider("Batch Size", 16, 256, 64)

if st.button("Start Training"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DCJDAN.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    ce_loss = nn.CrossEntropyLoss()

    source_data = load_uploaded_csv_as_tensor_dataset(source_file) if source_file else create_dummy_dataset(3000, 3)
    target_data = load_uploaded_csv_as_tensor_dataset(target_file) if target_file else create_dummy_dataset(1500, 3)
    source_loader = DataLoader(source_data, batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(target_data, batch_size=batch_size, shuffle=True)

    loss_values = []
    loss_plot = st.empty()
    tsne_plot = st.empty()
    progress_bar = st.progress(0)
    log_area = st.empty()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        f_s_all, f_t_all = [], []

        for (x_s, y_s), (x_t, _) in zip(source_loader, target_loader):
            x_s, y_s = x_s.to(device), y_s.to(device)
            x_t = x_t.to(device)

            y_s_pred, y_t_pred, f_s, f_t = model(x_s, x_t)
            pseudo_labels_t = torch.argmax(y_t_pred.detach(), dim=1)

            loss_cls = ce_loss(y_s_pred, y_s)
            loss_mmd = mmd_loss(f_s, f_t, y_s, pseudo_labels_t)
            loss_reg = l2_reg_loss(model.source_net, model.target_net)
            loss = loss_cls + 0.6 * loss_mmd + 0.001 * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            f_s_all.append(f_s.detach().cpu())
            f_t_all.append(f_t.detach().cpu())

        avg_loss = epoch_loss / len(source_loader)
        loss_values.append(avg_loss)

        with loss_plot.container():
            plt.figure()
            plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o')
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            st.pyplot(plt)

        if epoch % 5 == 0:
            with tsne_plot.container():
                f_s_cat = torch.cat(f_s_all)
                f_t_cat = torch.cat(f_t_all)
                labels = torch.cat([
                    torch.zeros(len(f_s_cat)),
                    torch.ones(len(f_t_cat))
                ])
                combined_features = torch.cat([f_s_cat, f_t_cat])
                reduced = TSNE(n_components=2).fit_transform(combined_features)
                plt.figure()
                plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='coolwarm', s=10)
                plt.title(f"t-SNE at Epoch {epoch+1}")
                st.pyplot(plt)

        progress_bar.progress((epoch + 1) / epochs)
        log_area.text(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    st.success("Training completed.")
