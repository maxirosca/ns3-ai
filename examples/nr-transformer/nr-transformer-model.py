import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NrTransformerModel(nn.Module):
    def __init__(self, input_dim=6, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.forward_layer = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        weights = self.forward_layer(x).squeeze(-1) 
        return weights
    

class NrDataset(Dataset):
    def __init__(self, csv_file, feature_scaler=None, label_scaler=None):
        self.data = pd.read_csv(csv_file)

        feature_cols = ['qci', 'priority', 'hol_delay', 'delay_budget', 'avg_tput', 'potential_tput']
        label_col = 'weight'

        # Group by scheduler_round (we assume each round has exactly 2 UEs)
        grouped = self.data.groupby('schedule_round')
        self.raw_features = []
        self.raw_labels = []

        for _, group in grouped:
            if len(group) != 2:
                continue  # skip incomplete rounds

            features = group[feature_cols].astype('float32').values
            labels = group[label_col].astype('float32').values.reshape(-1, 1)

            self.raw_features.append(features)  # shape: (2, 6)
            self.raw_labels.append(labels)      # shape: (2, 1)

        # Apply scaling
        if feature_scaler is None:
            self.feature_scaler = MinMaxScaler()
            all_features = np.vstack(self.raw_features)
            self.feature_scaler.fit(all_features)
        else:
            self.feature_scaler = feature_scaler

        if label_scaler is None:
            self.label_scaler = MinMaxScaler()
            all_labels = np.vstack(self.raw_labels)
            self.label_scaler.fit(all_labels)
        else:
            self.label_scaler = label_scaler

        # Transform
        self.features = [self.feature_scaler.transform(f) for f in self.raw_features]  # list of (2, 6)
        self.labels = [self.label_scaler.transform(l).squeeze(-1) for l in self.raw_labels]  # list of (2,)

        # Convert to tensors
        self.features = torch.tensor(np.stack(self.features), dtype=torch.float32)  # (N, 2, 6)
        self.labels = torch.tensor(np.stack(self.labels), dtype=torch.float32)      # (N, 2)

        print("Feature tensor shape:", self.features.shape)
        print("Label tensor shape:", self.labels.shape)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def get_scalers(self):
        return self.feature_scaler, self.label_scaler


csv_dataset = '/home/maximilianrosca/ns-3-dev/dl_weights_log_sample.csv'
dataset = NrDataset(csv_dataset)

indices = list(range(len(dataset)))
train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Initialize model, loss function, and optimizer
model = NrTransformerModel(input_dim=6).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(train_loader, model, criterion, optimizer):
    size = len(train_loader.dataset)  
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        # Forward pass
        pred = model(X)
        loss = criterion(pred, y)  # Ensure shape match
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(f"X shape: {X.shape}, y shape: {y.shape}, pred shape: {pred.shape}")

        if batch % 100 == 0 or batch == 0:
            current = batch * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

def validate(val_loader, model, criterion, label_scaler):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)  
            pred = model(X)

            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())
            loss = criterion(pred, y)
            val_loss += loss.item() * X.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    preds_org = label_scaler.inverse_transform(all_preds)
    labels_org = label_scaler.inverse_transform(all_labels)

    mae = np.mean(np.abs(preds_org - labels_org))
    print(f"Validation Loss (normalized): {avg_val_loss:.4f}")
    print(f"MAE (original scale): {mae:.2f}")
    return avg_val_loss

# Training loop
num_epochs = 5
for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, criterion, optimizer)
    validate(val_loader, model, criterion, dataset.get_scalers()[1])
    print("Done!")