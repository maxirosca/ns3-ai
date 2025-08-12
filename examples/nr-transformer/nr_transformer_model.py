import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import math


class NrTransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=1, num_layers=1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.forward_layer = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print("Input shape:", x.shape)
        # print("Input data:", x)
        x = self.embedding(x)
        # print("Input shape after embedding:", x.shape)
        # print("Input data after embedding:", x)
        x = self.positional_encoding(x)
        # print("Input shape after positional encoding:", x.shape)
        # print("Input after positional encoding:", x)  
        x = self.transformer(x)
        # print("Input shape after transformer:", x.shape)
        # print("Input after transformer:", x)
        x = self.forward_layer(x).squeeze(-1)
        # print("Input shape after FFN:", x.shape)
        # print("Input after FFN:", x)
        weights = self.softmax(x)
        # print("Output shape after softmax:", weights.shape)
        # print("Output after softmax:", weights)
        return weights
    

class NrDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        feature_cols = ['qci', 'priority', 'hol_delay', 'delay_budget', 'avg_tput', 'potential_tput']
        label_col = 'weight'

        # Group by total_schedule_round (we assume each round has exactly 2 UEs)
        grouped = self.data.groupby('total_schedule_round')
        self.raw_features = []
        self.raw_labels = []

        for _, group in grouped:
            if len(group) != 2:
                continue  # skip incomplete rounds

            features = group[feature_cols].astype('float32').values
            labels = group[label_col].astype('float32').values.reshape(-1, 1)

            self.raw_features.append(features)  # shape: (2, 6)
            self.raw_labels.append(labels)      # shape: (2, 1)
            
        # Convert to tensors
        self.features = torch.tensor(np.stack(self.raw_features), dtype=torch.float32)  # (N, 2, 6)
        self.labels = torch.tensor(np.stack(self.raw_labels), dtype=torch.float32).squeeze(-1)  # (N, 2)

        # print("Feature tensor shape:", self.features.shape)
        # print("Label tensor shape:", self.labels.shape)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)