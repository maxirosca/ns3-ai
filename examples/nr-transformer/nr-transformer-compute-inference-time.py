import time
import torch
import csv
import os
from nr_transformer_model import NrTransformerModel
from itertools import product
import numpy as np
import pandas as pd

# torch.manual_seed(0)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(0)

# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

# output_csv = "model_inference_times_and_parameters_new.csv"

# # Create file + header once
# if not os.path.exists(output_csv):
#     with open(output_csv, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "d_model",
#             "nhead",
#             "num_layers",
#             "avg_inference_time_ms",
#             "p95_inference_time_ms",
#             "p99_inference_time_ms",
#             "num_parameters"
#         ])

# def measure_inference_time(model, input, pad_mask, device, num_warmup=50, num_runs=200):
#     model.eval()
#     input = input.to(device)
#     pad_mask = pad_mask.to(device)

#     # Warm-up runs
#     with torch.no_grad():
#         for _ in range(num_warmup):
#             _ = model(input, src_key_padding_mask=pad_mask)
    
#     times_ms = []

#     # Timed runs
#     with torch.no_grad():
#         for _ in range(num_runs):
#             if device == "cuda":
#                 torch.cuda.synchronize()
#             start = time.perf_counter()
#             _ = model(input, src_key_padding_mask=pad_mask)
#             if device == "cuda":
#                 torch.cuda.synchronize()
#             end = time.perf_counter()
#             times_ms.append((end - start) * 1000)

#     times_ms = np.array(times_ms)
#     avg_time_ms = times_ms.mean()
#     p95_time_ms = np.percentile(times_ms, 95)
#     p99_time_ms = np.percentile(times_ms, 99)
#     return avg_time_ms, p95_time_ms, p99_time_ms

# hyperparameters_grid = {
#     "d_model": [16, 32, 64, 128, 256, 512],
#     "nhead": [1, 2, 4, 8],
#     "num_layers": [1, 2, 3, 4, 5, 6]
# }

# def generate_combinations(grid):
#     keys = list(grid.keys())
#     values = list(grid.values())
#     for combination in product(*values):
#         yield dict(zip(keys, combination))

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")

# df = pd.read_csv("~/Downloads/inputs_DlTransmission_zscore4.csv")
# slot_keys = df[["simulation", "frame", "subframe", "slot"]].drop_duplicates()
# random_key = slot_keys.sample(n=1, random_state=0).iloc[0]
# df_filtered = df[
#     (df["simulation"] == random_key["simulation"]) &
#     (df["frame"] == random_key["frame"]) &
#     (df["subframe"] == random_key["subframe"]) &
#     (df["slot"] == random_key["slot"])
# ]
# input_tensor = torch.tensor(df_filtered[[col for col in df_filtered.columns if col not in ["simulation", "frame", "subframe", "slot"]]].values, dtype=torch.float32).unsqueeze(0)

# pad_mask = (input_tensor[..., 0] == 0)

# for hparams in generate_combinations(hyperparameters_grid):
#     config_tuple = (
#         hparams["d_model"],
#         hparams["nhead"],
#         hparams["num_layers"]
#     )
#     print(f"Calculating inference time with hyperparameters: {hparams}")
#     model = NrTransformerModel(
#         d_model=hparams["d_model"],
#         nhead=hparams["nhead"],
#         num_layers=hparams["num_layers"]
#     ).to(device)

#     for p in model.parameters():
#         p.requires_grad = False

#     avg_time_ms, p95_time_ms, p99_time_ms = measure_inference_time(
#         model,
#         input=input_tensor,
#         pad_mask=pad_mask,
#         device=device,
#         num_warmup=50,
#         num_runs=200
#     )

#     params = sum(p.numel() for p in model.parameters())
#     with open(output_csv, "a", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             hparams["d_model"],
#             hparams["nhead"],
#             hparams["num_layers"],
#             f"{avg_time_ms:.3f}",
#             f"{p95_time_ms:.3f}",
#             f"{p99_time_ms:.3f}",
#             f"{params / 1e3:.2f}K"
#         ])
    
#     del model
#     if device == "cuda":
#         torch.cuda.empty_cache()

# df_output = pd.read_csv(output_csv)
# df_output = df_output.sort_values(by=["p99_inference_time_ms"], ascending=True)
# df_output.to_csv(output_csv, index=False)

# df_transformer_inference = pd.read_csv('scheduler_time_only_model_inference.csv')
# df_scheduler_inference = pd.read_csv('scheduler_time_dl.csv')

# numpy_array_transformer = df_transformer_inference.iloc[:, 0].to_numpy()
# numpy_array_scheduler_inference = df_scheduler_inference["duration"].to_numpy() 

# avg_time_transformer = numpy_array_transformer.mean()
# p95_time_transformer = np.percentile(numpy_array_transformer, 95)
# p99_time_transformer = np.percentile(numpy_array_transformer, 99)

# avg_time_scheduler = numpy_array_scheduler_inference.mean()
# p95_time_scheduler = np.percentile(numpy_array_scheduler_inference, 95)
# p99_time_scheduler = np.percentile(numpy_array_scheduler_inference, 99)

# print("Transformer Inference Times (ms):")
# print(f"Average: {avg_time_transformer:.4f} ms")
# print(f"95th Percentile: {p95_time_transformer:.4f} ms")
# print(f"99th Percentile: {p99_time_transformer:.4f} ms")
# print("\nScheduler Inference Times (ms):")
# print(f"Average: {avg_time_scheduler / 1e03:.4f} ms")
# print(f"95th Percentile: {p95_time_scheduler / 1e03:.4f} ms")
# print(f"99th Percentile: {p99_time_scheduler / 1e03:.4f} ms")

df_scheduler_inference = pd.read_csv('/home/maximilianrosca/Desktop/scheduler_time_dl.csv')
grouped = df_scheduler_inference.groupby('simulation')['duration']
summary = grouped.agg(['mean', lambda x: x.quantile(0.95), lambda x: x.quantile(0.99)])
summary.columns = ['avg_duration', 'p95_duration', 'p99_duration']
print("Final results - Scheduler Inference Times (ms):")
print(f"Average: {summary['avg_duration'].mean() / 1e03:.4f} ms")
print(f"95th Percentile: {summary['p95_duration'].mean() / 1e03:.4f} ms")
print(f"99th Percentile: {summary['p99_duration'].mean() / 1e03:.4f} ms")

print("Standard Deviation - Scheduler Inference Times (ms):")
print(f"Average: {summary['avg_duration'].std() / 1e03:.4f} ms")
print(f"95th Percentile: {summary['p95_duration'].std() / 1e03:.4f} ms")
print(f"99th Percentile: {summary['p99_duration'].std() / 1e03:.4f} ms")

df_transformer_inference = pd.read_csv('/home/maximilianrosca/Desktop/scheduler_time_only_model_inference.csv')

# Get the inference time column and divide into 10 sections
inference_times = df_transformer_inference['inference_time_ms'].values
section_size = len(inference_times) // 10
section_avg = {} 
for i in range(10):
    start_idx = i * section_size
    end_idx = start_idx + section_size if i < 9 else len(inference_times)
    section_avg[i] = inference_times[start_idx:end_idx].mean()
    print(f"Section {i+1} Average: {section_avg[i]:.4f} ms")
print("Overall Average Inference Time: {:.4f} ms".format(np.mean(list(section_avg.values()))))