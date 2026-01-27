import torch
import torch.nn as nn

class NrBaselineModel(nn.Module):
    def __init__(self, input_size=35, hidden_dim=128, output_size=12, num_ues=3):
        super().__init__()
        self.num_ues = num_ues + 1 # including rnti == 0
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, output_size * self.num_ues)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        out = self.net(x)
        out = out.view(-1, self.output_size, self.num_ues)
        return out

model = NrBaselineModel(input_size=35, hidden_dim=240, output_size=12, num_ues=3)
pytorch_total_params = sum(p.numel() for p in model.parameters())  
print(f"Total parameters: {pytorch_total_params}")
for name, param in model.named_parameters():
    print(f"{name:40s} {param.numel():6d}")