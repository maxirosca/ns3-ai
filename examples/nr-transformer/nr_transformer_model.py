import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class NrTransformerModel(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, num_ues=4):
        super().__init__()
        self.num_ues = num_ues
        self.embedding = nn.Linear(7, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.forward_layer = nn.Sequential(
            nn.Linear(5 * d_model, 400),
            nn.ReLU(),
	    nn.Dropout(0.3),
            nn.Linear(400, 12 * num_ues)
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
             "available_symbols", "queue_size", "mcs"]
        out_sym_cols = [f"sym{i}" for i in range(1, 13)]

        flow_counts = self.input_data.groupby(slot_cols).size() # Drop the ~200 inputs with 6 features instead of 5
        valid_slots = flow_counts[flow_counts == 5].index # Drop the ~200 inputs with 6 features instead of 5
        self.input_data = self.input_data[self.input_data.set_index(slot_cols).index.isin(valid_slots)] # Drop the ~200 inputs with 6 features instead of 5
        self.output_data = self.output_data[self.output_data.set_index(slot_cols).index.isin(valid_slots)] # Drop the ~200 inputs with 6 features instead of 5
        self.output_data = self.output_data.drop_duplicates(subset=slot_cols, keep='first') # Drop the ~2000 duplicates from simulation 121
        self.output_indexed = self.output_data.set_index(slot_cols, drop=False)
        
        self.raw_features = []
        self.raw_labels = []

        for key, group in self.input_data.groupby(slot_cols):
            features = group[features_cols].astype('float32').values
            self.raw_features.append(features)  

            label_row = self.output_indexed.loc[key]
            labels = label_row[out_sym_cols].astype('float32').values.flatten()
            self.raw_labels.append(labels)

        for i, f in enumerate(self.raw_features):
            if f.shape != (5, 7):
                print(f"Inconsistent feature shape at index {i}: {f.shape}")

        for i, l in enumerate(self.raw_labels):
            if l.shape != (12,):
                print(f"Inconsistent label shape at index {i}: {l.shape}")
        print("Features:", len(self.raw_features), "Labels:", len(self.raw_labels))

        # Convert to tensors
        self.features = torch.tensor(np.stack(self.raw_features), dtype=torch.float32) # torch.size([5, 7])
        self.labels = torch.tensor(np.stack(self.raw_labels), dtype=torch.long)  # torch.size([12])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
pytorch_total_params = sum(p.numel() for p in NrTransformerModel().parameters())
pytorch_total_trainable_params = sum(p.numel() for p in NrTransformerModel().parameters() if p.requires_grad)
print(f"Total parameters: {pytorch_total_params}")
print(f"Total trainable parameters: {pytorch_total_trainable_params}")
