import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.dcjdan_model import DCJDAN
from utils import create_dummy_dataset, l2_reg_loss, mmd_loss

def train_dcjdan(epochs=None, lr=None, batch_size=None, source_data=None, target_data=None):
    # Default values
    epochs = epochs or 10
    lr = lr or 0.001
    batch_size = batch_size or 64

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DCJDAN(num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    source_loader = DataLoader(
        source_data if source_data else create_dummy_dataset(3000, 3),
        batch_size=batch_size, shuffle=True
    )
    target_loader = DataLoader(
        target_data if target_data else create_dummy_dataset(1500, 3),
        batch_size=batch_size, shuffle=True
    )

    loss_values = []
    features_per_epoch = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
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

        avg_loss = epoch_loss / len(source_loader)
        loss_values.append(avg_loss)

        # Collect features for t-SNE visualization
        features_per_epoch.append((f_s.detach().cpu(), f_t.detach().cpu()))

    return loss_values, features_per_epoch
