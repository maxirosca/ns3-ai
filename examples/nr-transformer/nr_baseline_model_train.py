from pyexpat import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from nr_transformer_model import NrTransformerModel, NrDataset
import csv
from datetime import datetime
from itertools import product
import os
import pandas as pd
import numpy as np
from nr_baseline_model import NrBaselineModel


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

log_file_training_validation = 'training_validation_log_file_gridsearch_baselinemodel.csv'
try:
    with open(log_file_training_validation, 'x', newline="") as f:
        writer = csv.writer(f)
        header = ["timestamp", "lr", "batch_size", "epoch", "train_loss", "train_acc_symbol", "train_acc_slot", "val_loss", "val_acc_symbol", "val_acc_slot"]
        writer.writerow(header)
except FileExistsError:
    pass  # File already exists

if os.path.exists(log_file_training_validation):
    df = pd.read_csv(log_file_training_validation)
    trained_configs = set(
        tuple(row)
        for row in df[["lr", "batch_size"]].values
    )
    print(f"Loaded {len(trained_configs)} trained configurations")
else:
    trained_configs = set()
    print("No privrous training logs found")

log_file_test = 'test_log_file_gridsearch_baselinemodel.csv'
try:
    with open(log_file_test, 'x', newline="") as f:
        writer = csv.writer(f)
        header = ["timestamp", "lr", "batch_size",
                  "test_loss", "test_acc_symbol", "test_acc_slot"]
        writer.writerow(header)
except FileExistsError:
    pass  # File already exists

csv_input_file = '~/maxi_model_training/training_dataset4.1/inputs_DlTransmission_zscore.csv'
csv_output_file = '~/maxi_model_training/training_dataset4.1/outputs_DlTransmission_no_duplicates.csv'
dataset = NrDataset(csv_input_file, csv_output_file)

hyperparameters_grid = {
    "lr": [1e-5],
    "batch_size": [16, 32, 64]
}

def generate_combinations(grid):
    keys = list(grid.keys())
    values = list(grid.values())
    for combination in product(*values):
        yield dict(zip(keys, combination))

num_samples = len(dataset)
indices = list(range(num_samples))
train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=42, shuffle=True)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, shuffle=True)

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

def run_training(hparams):
    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size= 32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size= 32, shuffle=False)

 

    # Initialize model, loss function, and optimizer
    model = NrBaselineModel(input_size=35, hidden_dim=128, output_size=12, num_ues=3).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])

    # Training loop
    best_val_loss = float('inf')
    patience, patience_counter = 20, 0
    num_epochs = 300
    model_path = (
         f"best-model-params.2/"
         f"nr_baseline_model_best_lr{hparams['lr']}_bs{hparams['batch_size']}.pth"
    )
    model_name = (
         f"nr_baseline_model_best_lr{hparams['lr']}_bs{hparams['batch_size']}.pth"
    )
    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss, train_acc_symbol, train_acc_slot = train(train_loader, model, loss_fn, optimizer)
        val_loss, val_acc_symbol, val_acc_slot = validation(val_loader, model, loss_fn)
        log_training_validation(t, train_loss, train_acc_symbol, train_acc_slot, val_loss, val_acc_symbol, val_acc_slot, hparams)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print("New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Model testing
    print("Testing the best model on the test set!")
    model.load_state_dict(torch.load(model_path))
    test_loss, test_acc_symbol, test_acc_slot = test(test_loader, model, loss_fn)
    log_test(test_loss, test_acc_symbol, test_acc_slot, hparams)
    print("Testing completed!")

def train(train_loader, model, loss_fn, optimizer):
    total_loss = 0.0
    total_correct_symbols = 0
    total_symbols = 0
    total_correct_slots = 0
    total_slots = 0
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        X_flat = X.view(X.size(0), -1)
        
        # Forward pass and compute loss
        pred = model(X_flat)
        # print(f"Model output shape before permute: {pred.shape}")
        pred = pred.permute(0, 2, 1)
        # print(f"Model output shape after permute: {pred.shape}")
        loss = loss_fn(pred, y)
        predicted_classes = pred.argmax(dim=1)

        # Per-symbol
        correct_symbol = (predicted_classes == y)
        total_correct_symbols += correct_symbol.sum().item()
        total_symbols += y.numel()

        # Per-slot
        correct_slot = (predicted_classes == y).all(dim=1)
        total_correct_slots += correct_slot.sum().item()
        total_slots += y.size(0)

        # Backprop and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    acc_symbol = total_correct_symbols / total_symbols
    acc_slot = total_correct_slots / total_slots
    print(f"Train Loss: {avg_loss:.4f} | Train Acc/Sym: {acc_symbol*100:.2f}% | Train Acc/Slot: {acc_slot*100:.2f}%")
    return avg_loss, acc_symbol, acc_slot

def validation(val_loader, model, loss_fn):
    total_loss = 0.0
    total_correct_symbols = 0
    total_symbols = 0
    total_correct_slots = 0
    total_slots = 0
    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            X_flat = X.view(X.size(0), -1)
            pred = model(X_flat)
            pred = pred.permute(0, 2, 1)
            loss = loss_fn(pred, y)
            predicted_classes = pred.argmax(dim=1)
            total_loss += loss.item()

            # Per-symbol
            correct_symbol = (predicted_classes == y)
            total_correct_symbols += correct_symbol.sum().item()
            total_symbols += y.numel()

            # Per-slot
            correct_slot = (predicted_classes == y).all(dim=1)
            total_correct_slots += correct_slot.sum().item()
            total_slots += y.size(0)

    avg_loss = total_loss / len(val_loader)
    acc_symbol = total_correct_symbols / total_symbols
    acc_slot = total_correct_slots / total_slots
    print(f"Validation Loss: {avg_loss:.4f} | Validation Acc/Sym: {acc_symbol*100:.2f}% | Validation Acc/Slot: {acc_slot*100:.2f}%")
    return avg_loss, acc_symbol, acc_slot

def test(test_loader, model, loss_fn):
    total_loss = 0.0
    total_correct_symbols = 0
    total_symbols = 0
    total_correct_slots = 0
    total_slots = 0
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            X_flat = X.view(X.size(0), -1)
            pred = model(X_flat)
            pred = pred.permute(0, 2, 1)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            predicted_classes = pred.argmax(dim=1)

            # Per-symbol
            correct_symbol = (predicted_classes == y)
            total_correct_symbols += correct_symbol.sum().item()
            total_symbols += y.numel()

            # Per-slot
            correct_slot = (predicted_classes == y).all(dim=1)
            total_correct_slots += correct_slot.sum().item()
            total_slots += y.size(0)

    avg_loss = total_loss / len(test_loader)
    acc_symbol = total_correct_symbols / total_symbols
    acc_slot = total_correct_slots / total_slots
    print(f"Test Loss: {avg_loss:.4f} | Test Acc/Sym: {acc_symbol*100:.2f}% | Test Acc/Slot: {acc_slot*100:.2f}%")
    return avg_loss, acc_symbol, acc_slot

def log_training_validation(epoch, train_loss, train_acc_symbol, train_acc_slot, val_loss, val_acc_symbol, val_acc_slot, hparams):
    with open(log_file_training_validation, "a", newline="") as f:
        writer = csv.writer(f)
        row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               hparams["lr"],
               hparams["batch_size"],
               epoch + 1,
               round(train_loss, 4),
               round(train_acc_symbol*100, 2),
               round(train_acc_slot*100, 2),
               round(val_loss, 4),
               round(val_acc_symbol*100, 2),
               round(val_acc_slot*100, 2)]
        writer.writerow(row)

def log_test(test_loss, test_acc_symbol, test_acc_slot, hparams):
    with open(log_file_test, "a", newline="") as f:
        writer = csv.writer(f)
        row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               hparams["lr"],
               hparams["batch_size"],
               round(test_loss, 4),
               round(test_acc_symbol*100, 2),
               round(test_acc_slot*100, 2)]
        writer.writerow(row)

for hparams in generate_combinations(hyperparameters_grid):
    config_tuple = (
        hparams["lr"],
        hparams["batch_size"]
    )
    if config_tuple in trained_configs:
        print(f"Skipping already trained configuration: {hparams}")
        continue
    print(f"Running training with hyperparameters: {hparams}")
    run_training(hparams)

