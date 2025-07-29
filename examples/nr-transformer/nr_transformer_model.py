import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


class NrTransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
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

        # print("Feature tensor shape:", self.features.shape)
        # print("Label tensor shape:", self.labels.shape)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def get_scalers(self):
        return self.feature_scaler, self.label_scaler