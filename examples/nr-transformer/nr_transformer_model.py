import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class NrTransformerModel(nn.Module):
    def __init__(self, d_model=16, nhead=2, num_layers=1, num_ues=4):
        super().__init__()
        self.num_ues = num_ues
        self.embedding = nn.Linear(8, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.forward_layer = nn.Sequential(
            nn.Linear(5 * d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 12 * num_ues)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        x = self.forward_layer(x)
        # print(f"Model output shape before reshape: {x.shape}")
        x = x.view(-1, 12, self.num_ues)
        # print(f"Model output shape after reshape: {x.shape}")
        return x


class NrDataset(Dataset):
    def __init__(self, csv_input_file, csv_output_file):
        self.input_data = pd.read_csv(csv_input_file)
        self.output_data = pd.read_csv(csv_output_file)

        slot_cols = ['simulation', 'frame', 'subframe', 'slot']
        features_cols = ["rnti", "resource_type", "priority", "delay_budget",
             "hol_delay", "available_symbols", "queue_size", "mcs"]
        out_sym_cols = [f"sym{i}" for i in range(1, 13)]

        self.output_indexed = self.output_data.set_index(slot_cols, drop=False)
        
        self.raw_features = []
        self.raw_labels = []

        for key, group in self.input_data.groupby(slot_cols):
            features = group[features_cols].astype('float32').values
            self.raw_features.append(features)  

            label_row = self.output_indexed.loc[key]
            labels = label_row[out_sym_cols].astype('float32').values
            self.raw_labels.append(labels)
 
        # Convert to tensors
        self.features = torch.tensor(np.stack(self.raw_features), dtype=torch.float32) # torch.size([5, 8])
        self.labels = torch.tensor(np.stack(self.raw_labels), dtype=torch.long)  # torch.size([12])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]