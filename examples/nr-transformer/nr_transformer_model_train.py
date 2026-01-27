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


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

log_file_training_validation = '~/maxi_model_training/ns3-ai/examples/nr-transformer/gridsearch.2/training_validation_gridsearch_combined1-7.csv'
try:
    with open(log_file_training_validation, 'x', newline="") as f:
        writer = csv.writer(f)
        header = ["timestamp", "lr", "d_model", "nhead", "num_layers", "batch_size",
                  "epoch", "train_loss", "train_acc_symbol", "train_acc_slot", "val_loss", "val_acc_symbol", "val_acc_slot"]
        writer.writerow(header)
except FileExistsError:
    pass  # File already exists

if os.path.exists(log_file_training_validation):
    df = pd.read_csv(log_file_training_validation)
    trained_configs = set(
        tuple(row)
        for row in df[["lr", "d_model", "nhead", "num_layers", "batch_size"]].values
    )
    print(f"Loaded {len(trained_configs)} trained configurations")
else:
    trained_configs = set()
    print("No privrous training logs found")

log_file_test = 'test_log_file-v2.csv'
try:
    with open(log_file_test, 'x', newline="") as f:
        writer = csv.writer(f)
        header = ["timestamp", "lr", "d_model", "nhead", "num_layers", "batch_size",
                  "test_loss", "test_acc_symbol", "test_acc_slot"]
        writer.writerow(header)
except FileExistsError:
    pass  # File already exists

# log_file_inference = 'inference_log_file.csv'
# with open(log_file_inference, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow([
#         "inference_id",
#         *[f"y_true_{i}" for i in range(12)],
#         *[f"y_pred_{i}" for i in range(12)]
#     ])

csv_input_file = '~/maxi_model_training/training_dataset4.1/inputs_DlTransmission_zscore.csv'
csv_output_file = '~/maxi_model_training/training_dataset4.1/outputs_DlTransmission_no_duplicates.csv'
dataset = NrDataset(csv_input_file, csv_output_file)

# --------------------------------------------------------------------------------->
# ---------------------------------- Split dataset in 3 sets ---------------------->
# --------------------------------------------------------------------------------->
N = len(dataset)
indices_split = np.arange(N)

# Shuffle once
rng = np.random.default_rng(seed=42)
rng.shuffle(indices_split)

# Define sizes
size_10 = int(0.1 * N)
size_33 = int(0.33 * N)
size_66 = int(0.66 * N)

# Nested subsets
idx_10 = indices_split[:size_10]
idx_33 = indices_split[:size_33]
idx_66 = indices_split[:size_66]

# Create the actual datasets
dataset_10 = Subset(dataset, idx_10)
dataset_33 = Subset(dataset, idx_33)
dataset_66 = Subset(dataset, idx_66)
# --------------------------------------------------------------------------------->
# ---------------------------------- End split dataset in 3 sets ------------------>
# --------------------------------------------------------------------------------->

hyperparameters_grid = {
    "lr": [1e-5],
    "d_model": [16],
    "nhead": [1, 2],
    "num_layers": [1, 2, 3, 4, 5, 6],
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

    # for xb, yb in train_loader:
    #     print("Input batch:", xb.shape)  # e.g. [32, 5, 7]
    #     print("Output batch:", yb.shape) # e.g. [32, 12]
    #     break

    # Initialize model, loss function, and optimizer
    # model_params = torch.load('/home/maximilianrosca/ns-3-dev/contrib/ai/examples/nr-transformer/nr_transformer_model.pth')
    model = NrTransformerModel(d_model=hparams["d_model"], nhead=hparams["nhead"], num_layers=hparams["num_layers"]).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())  
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {pytorch_total_params}")
    print(f"Total trainable parameters: {pytorch_total_trainable_params}")
    # model.load_state_dict(model_params['model_state_dict'])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])
    # optimizer.load_state_dict(model_params['optimizer_state_dict'])
     
    # Training loop
    best_val_loss = float('inf')
    # best_val_loss = model_params['val_loss']
    patience, patience_counter = 20, 0
    num_epochs = 300
    # num_epochs =  model_params['epoch']
    model_path = (
         f"best-model-params.2/"
         f"nr_transformer_model_best_lr{hparams['lr']}_dmodel{hparams['d_model']}"
         f"_nhead{hparams['nhead']}_layers{hparams['num_layers']}"
         f"_bs{hparams['batch_size']}.pth"
    )
    model_name = (
         f"nr_transformer_model_best_lr{hparams['lr']}_dmodel{hparams['d_model']}"
         f"_nhead{hparams['nhead']}_layers{hparams['num_layers']}"
         f"_bs{hparams['batch_size']}.pth"
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
        # Forward pass and compute loss
        pad_mask = (X[..., 0] == 0)
        pred = model(X, src_key_padding_mask=pad_mask)
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
            pad_mask = (X[..., 0] == 0)
            pred = model(X, src_key_padding_mask=pad_mask)
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
    #inference_id = 0
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pad_mask = (X[..., 0] == 0)
            pred = model(X, src_key_padding_mask=pad_mask)
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

            # # Move to CPU once
            # y_cpu = y.cpu().numpy()
            # preds_cpu = predicted_classes.cpu().numpy()
            # with open(log_file_inference, "a", newline="") as f:
            #     writer = csv.writer(f)
            #     for i in range(y_cpu.shape[0]):
            #         writer.writerow([
            #             inference_id,
            #             *y_cpu[i].tolist(),
            #             *preds_cpu[i].tolist()
            #         ])
            #         inference_id += 1

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
               hparams["d_model"],
               hparams["nhead"],
               hparams["num_layers"],
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
               hparams["d_model"],
               hparams["nhead"],
               hparams["num_layers"],
               hparams["batch_size"],
               round(test_loss, 4),
               round(test_acc_symbol*100, 2),
               round(test_acc_slot*100, 2)]
        writer.writerow(row)

for hparams in generate_combinations(hyperparameters_grid):
    config_tuple = (
        hparams["lr"],
        hparams["d_model"],
        hparams["nhead"],
        hparams["num_layers"],
        hparams["batch_size"]
    )
    if config_tuple in trained_configs:
        print(f"Skipping already trained configuration: {hparams}")
        continue
    print(f"Running training with hyperparameters: {hparams}")
    run_training(hparams)

# hparams = {
#     "lr": 0.0005,
#     "d_model": 64,
#     "nhead": 8,
#     "num_layers": 2,
#     "batch_size": 32
# }
# run_training(hparams)

# # ----------------------------------------------------------
# test_loader = DataLoader(test_dataset, batch_size= 32, shuffle=False)
# model = NrTransformerModel(d_model=16, nhead=2, num_layers=1, num_ues=4).to(device)
# model.load_state_dict(torch.load('nr_transformer_model_best_lr0.0001_dmodel16_nhead2_layers1_bs32.pth'))
# loss_fn = nn.CrossEntropyLoss()
# test_loss, test_acc = test(test_loader, model, loss_fn)
# # ----------------------------------------------------------
