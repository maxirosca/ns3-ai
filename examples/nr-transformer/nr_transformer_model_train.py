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


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

log_file_training_validation = 'training_validation_log_file.csv'
try:
    with open(log_file_training_validation, 'x', newline="") as f:
        writer = csv.writer(f)
        header = ["timestamp", "lr", "d_model", "nhead", "num_layers", "batch_size",
                  "epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
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

log_file_test = 'test_log_file.csv'
try:
    with open(log_file_test, 'x', newline="") as f:
        writer = csv.writer(f)
        header = ["timestamp", "lr", "d_model", "nhead", "num_layers", "batch_size",
                  "test_loss", "test_acc"]
        writer.writerow(header)
except FileExistsError:
    pass  # File already exists

csv_input_file = '~/maxi_model_training/training_dataset4/inputs_DlTransmission_zscore4.csv'
csv_output_file = '~/maxi_model_training/training_dataset4/outputs_DlTransmission_no_duplicates4.csv'
dataset = NrDataset(csv_input_file, csv_output_file)

# hyperparameters_grid = {
#     "lr": [5e-4, 3e-4, 1e-4, 5e-5],
#     "d_model": [64, 128, 256, 512],
#     "nhead": [2, 4, 8],
#     "num_layers": [2, 3, 4, 5, 6],
#     "batch_size": [16, 32, 64]
# }

# def generate_combinations(grid):
#     keys = list(grid.keys())
#     values = list(grid.values())
#     for combination in product(*values):
#         yield dict(zip(keys, combination))

num_samples = len(dataset)
indices = list(range(num_samples))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, shuffle=True)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.3, random_state=42, shuffle=True)

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
    # model.load_state_dict(model_params['model_state_dict'])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])
    # optimizer.load_state_dict(model_params['optimizer_state_dict'])
     
    # Training loop
    best_val_loss = float('inf')
    # best_val_loss = model_params['val_loss']
    # patience, patience_counter = 5, 0
    num_epochs = 5
    # model_name = (
    #     f"nr_transformer_model_best_lr{hparams['lr']}_dmodel{hparams['d_model']}"
    #     f"_nhead{hparams['nhead']}_layers{hparams['num_layers']}"
    #     f"_bs{hparams['batch_size']}.pth"
    # )
    # num_epochs = model_params['epoch']
    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss, train_acc = train(train_loader, model, loss_fn, optimizer)
        val_loss, val_acc = validation(val_loader, model, loss_fn)
        log_training_validation(t, train_loss, train_acc, val_loss, val_acc, hparams)
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     patience_counter = 0
        #     torch.save(model.state_dict(), model_name)
        #     print("New best model saved!")
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print("Early stopping triggered.")
        #         break
    
    # # Model testing
    # print("Testing the best model on the test set!")
    # model.load_state_dict(torch.load("nr_transformer_model_best.pth"))
    # test_loss, test_acc = test(test_loader, model, loss_fn, hparams)
    # log_test(test_loss, test_acc)
    # print("Testing completed!")

def train(train_loader, model, loss_fn, optimizer):
    total_loss = 0.0
    total_accuracy = 0.0
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        # Forward pass and compute loss
        pred = model(X)
        # print(f"Model output shape before permute: {pred.shape}")
        pred = pred.permute(0, 2, 1)
        # print(f"Model output shape after permute: {pred.shape}")
        loss = loss_fn(pred, y)
        predicted_classes = pred.argmax(dim=1)
        correct = (predicted_classes == y).float()
        accuracy = correct.mean().item()
        # Backprop and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_accuracy += accuracy

    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)
    print(f"Train Loss: {avg_loss:.4f} | Train Accuracy: {avg_accuracy*100:.2f}%")
    return avg_loss, avg_accuracy

def validation(val_loader, model, loss_fn):
    total_loss = 0.0
    total_accuracy = 0.0
    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = pred.permute(0, 2, 1)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            predicted_classes = pred.argmax(dim=1)
            correct = (predicted_classes == y).float()
            total_accuracy += correct.mean().item()

    avg_loss = total_loss / len(val_loader)
    avg_accuracy = total_accuracy / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f} | Validation Accuracy: {avg_accuracy*100:.2f}%")
    return avg_loss, avg_accuracy

def test(test_loader, model, loss_fn):
    total_loss = 0.0
    total_accuracy = 0.0
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = pred.permute(0, 2, 1)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            predicted_classes = pred.argmax(dim=1)
            correct = (predicted_classes == y).float()
            total_accuracy += correct.mean().item()

    avg_loss = total_loss / len(test_loader)
    avg_accuracy = total_accuracy / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {avg_accuracy*100:.2f}%")
    return avg_loss, avg_accuracy

def log_training_validation(epoch, train_loss, train_acc, val_loss, val_acc, hparams):
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
               round(train_acc*100, 2),
               round(val_loss, 4),
               round(val_acc*100, 2)]
        writer.writerow(row)

def log_test(test_loss, test_acc, hparams):
    with open(log_file_test, "a", newline="") as f:
        writer = csv.writer(f)
        row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               hparams["lr"],
               hparams["d_model"],
               hparams["nhead"],
               hparams["num_layers"],
               hparams["batch_size"],
               round(test_loss, 4),
               round(test_acc*100, 2)]
        writer.writerow(row)

# for hparams in generate_combinations(hyperparameters_grid):
#     config_tuple = (
#         hparams["lr"],
#         hparams["d_model"],
#         hparams["nhead"],
#         hparams["num_layers"],
#         hparams["batch_size"]
#     )
#     if config_tuple in trained_configs:
#         print(f"Skipping already trained configuration: {hparams}")
#         continue
#     print(f"Running training with hyperparameters: {hparams}")
#     run_training(hparams)

hparams = {
    "lr": 0.0005,
    "d_model": 64,
    "nhead": 8,
    "num_layers": 2,
    "batch_size": 32
}
run_training(hparams)