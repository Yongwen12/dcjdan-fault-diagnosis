# install requiremnents if not pre-installed
!pip install torch scikit-learn matplotlib seaborn umap-learn

# backbone
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBackbone(nn.Module):
    def __init__(self):
        super(CNNBackbone, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 64)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 3)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(32, 64, 3)
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(64, 64, 3)
        self.pool4 = nn.MaxPool1d(2)
        self.conv5 = nn.Conv1d(64, 64, 3)
        self.pool5 = nn.MaxPool1d(2)
        self.conv6 = nn.Conv1d(64, 64, 3)
        self.pool6 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()

        # Calculate the output size of the last pooling layer
        # Dummy input to calculate the shape after conv and pooling layers
        dummy_input = torch.randn(1, 1, 1024)
        x = self.pool1(F.relu(self.conv1(dummy_input)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.pool6(F.relu(self.conv6(x)))
        flattened_size = x.size(1) * x.size(2)

        self.fc1 = nn.Linear(flattened_size, 200)
        self.fc2 = nn.Linear(200, 100)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.pool6(F.relu(self.conv6(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DCJDAN(nn.Module):
    def __init__(self, num_classes=3):
        super(DCJDAN, self).__init__()
        self.source_net = CNNBackbone()
        self.target_net = CNNBackbone()
        self.classifier = nn.Linear(100, num_classes)

    def forward(self, x_s, x_t):
        f_s = self.source_net(x_s)
        f_t = self.target_net(x_t)
        y_s = self.classifier(f_s)
        y_t = self.classifier(f_t)
        return y_s, y_t, f_s, f_t

# Define Train

import torch.nn as nn
import torch.optim as optim

def train_dcjdan():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DCJDAN(num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()

    source_loader = DataLoader(create_dummy_dataset(3000, 3), batch_size=64, shuffle=True)
    target_loader = DataLoader(create_dummy_dataset(1500, 3), batch_size=64, shuffle=True)

    for epoch in range(5):
        model.train()
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

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Train
train_dcjdan()
