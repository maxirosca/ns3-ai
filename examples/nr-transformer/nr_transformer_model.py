import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class NrTransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_ues=3):
        super().__init__()
        self.num_ues = num_ues + 1 # including rnti == 0 for nonallocated symbols
        self.embedding = nn.Linear(8, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        d_ffn = 10 * d_model
        
        self.forward_layer = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_ffn, 12 * self.num_ues)
        )

    def masked_mean(self, x, pad_mask):
        valid_mask = (~pad_mask).unsqueeze(-1).float()
        summed = (x * valid_mask).sum(dim=1)
        counts = valid_mask.sum(dim=1).clamp(min=1)
        return summed / counts
    
    def forward(self, x, src_key_padding_mask=None):
        x = self.embedding(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            x = self.masked_mean(x, src_key_padding_mask)
        else:
            x = x.mean(dim=1)
        x = self.forward_layer(x)
        # print(f"Model output shape before reshape: {x.shape}")
        x = x.view(-1, 12, self.num_ues)
        # print(f"Model output shape after reshape: {x.shape}")
        return x
    
class NrDataset(Dataset):
    def __init__(self, input_data: pd.DataFrame, output_data: pd.DataFrame):
        super().__init__()
        
        slot_cols = ['simulation', 'frame', 'subframe', 'slot']
        features_cols = ["rnti", "resource_type", "priority", "delay_budget",
             "available_symbols", "queue_size", "mcs", "bandwidth"]
        out_sym_cols = [f"sym{i}" for i in range(1, 13)]

        # Ensure deterministic ordering
        input_data = input_data.sort_values(slot_cols).reset_index(drop=True)
        output_data = output_data.sort_values(slot_cols).reset_index(drop=True)

        # Sanity check
        assert len(input_data) % 5 == 0, "Input data rows not multiple of 5"
        assert len(output_data) * 5 == len(input_data), "Output data rows not matching input data rows"

        # Build tensors
        X = input_data[features_cols].values.astype(np.float32)
        Y = output_data[out_sym_cols].values.astype(np.int64)

        # Reshape
        self.features = torch.from_numpy(X).view(-1, 5, len(features_cols))
        self.labels = torch.from_numpy(Y)

        print(f"Loaded {len(self.features)} samples | "
              f"X shape: {self.features.shape}, Y shape: {self.labels.shape}"
              )
        
    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
