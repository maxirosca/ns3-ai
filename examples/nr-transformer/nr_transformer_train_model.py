import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from nr_transformer_model import NrTransformerModel, NrDataset
import numpy as np


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


csv_path = '/home/maximilianrosca/Masterarbeit/Simulation-files/training_data_zscore_softmax.csv'
dataset = NrDataset(csv_path)

train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.3, random_state=42)
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32, shuffle=False)


# # Initialize model, loss function, and optimizer
model = NrTransformerModel(input_dim=6).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(train_loader, model, loss_fn, optimizer):
    total_loss = 0  
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        # Forward pass and compute loss
        pred = model(X)
        loss = loss_fn(pred, y) 
        # Backprop and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)

    print(f"Average Training Loss: {total_loss / len(train_loader.dataset):.4f}")


def test(test_loader, model, loss_fn):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)  
            pred = model(X)

            loss = loss_fn(pred, y)
            val_loss += loss.item() * X.size(0)

    avg_val_loss = val_loss / len(test_loader.dataset)
    print(f"Test Loss: {avg_val_loss:.4f}")


# Training loop
num_epochs = 10
for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
    print("Done!")

# Save the model
torch.save(model.state_dict(), "/home/maximilianrosca/Masterarbeit/Simulation-files/nr_transformer_model_zscore_softmax.pth")
print("Model saved!")