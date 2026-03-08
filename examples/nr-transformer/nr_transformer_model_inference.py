import nr_transformer_interface_py_vec as py_binding
from ns3ai_utils import Experiment
import sys
import traceback
import torch
from nr_transformer_model import NrTransformerModel
import argparse
import csv
from datetime import datetime
import os
import time
import gc

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

FLOW_NUMS = 5
# output_csv = ".csv"

# # Create file + header once
# if not os.path.exists(output_csv):
#     with open(output_csv, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "inference_time_ms"
#         ])

parser = argparse.ArgumentParser()
parser.add_argument("--randomStream", type=int, required=True)
parser.add_argument("--bandwidth", type=int, required=True)
parser.add_argument("--scenario", type=str, required=True)
parser.add_argument("--enableTransformer", type=bool, required=True)
args = parser.parse_args()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NrTransformerModel(d_model=32, nhead=8, num_layers=1)
model.load_state_dict(torch.load('nr_transformer_model_best_lr0.0001_dmodel32_nhead8_layers1_bs64.pth', map_location=torch.device('cpu'), weights_only=True), strict=True)
model = model.to(device)
model.eval()
input_tensor = torch.tensor([
    [ 2.0000, -1.2684,  0.0000, -0.0082, -0.4588],
    [ 1.0000, -0.1782,  0.0000, -0.0082, -0.4588],
    [ 2.0000, -0.1782,  0.0000, -0.0082, -0.4588],
    [ 2.0000,  1.1844,  1.0000, -0.0082,  2.2508],
    [ 3.0000,  1.1844,  1.0000, -0.0082,  2.2508]
], dtype=torch.float32).unsqueeze(0).to(device)
pad_mask_tensor = (input_tensor[..., 0] == 0).to(device)

# Warm-up inference
with torch.no_grad():
    for _ in range(10):
        _ = model(input_tensor, src_key_padding_mask=pad_mask_tensor)

# with torch.no_grad():
#     assert not model.training, "Model is still in training mode!"
#     outputs1 = model(input_tensor, src_key_padding_mask=pad_mask_tensor).argmax(dim=-1)
#     outputs2 = model(input_tensor, src_key_padding_mask=pad_mask_tensor).argmax(dim=-1)
#     print("Δ:", torch.max(torch.abs(outputs1 - outputs2)))
#     print("Pred1:", outputs1)
#     print("Pred2:", outputs2)

norm_params = torch.load("normalization_params.pt", map_location=torch.device('cpu'))
mean = norm_params["mean"].to(device)
std = norm_params["std"].to(device)

assert mean.shape == (3,)
assert std.shape == (3,)

inputs = torch.zeros(1, FLOW_NUMS, 5, dtype=torch.float32, device=device)
inputs_raw = torch.zeros(1, FLOW_NUMS, 5, dtype=torch.float32, device=device)
pad_mask = torch.zeros((1, FLOW_NUMS), dtype=torch.bool, device=device)

gc.disable()

print("Starting nr_transformer_use_model.py...")
exp = Experiment("nr_transformer_demo", "../../../../", py_binding,
                handleFinish=True, useVector=True, vectorSize=FLOW_NUMS)
setting_map = {
    "enableTransformer": args.enableTransformer,
    "randomStream": args.randomStream,
    "bandwidth": args.bandwidth,
    "scenario": args.scenario
    }

msgInterface = exp.run(setting_map, show_output=True)
# msgInterface = exp.run(show_output=True)
print("Experiment started...")

try:
    while True:
        msgInterface.PyRecvBegin()
        if msgInterface.PyGetFinished():
            break

        msgInterface.PySendBegin()
        cpp_vec = msgInterface.GetCpp2PyVector()
        for i in range(FLOW_NUMS):
            v = cpp_vec[i]
            inputs_raw[0, i, 0] = v.rnti
            inputs_raw[0, i, 1] = v.priority
            inputs_raw[0, i, 2] = v.dcGbrFlag
            inputs_raw[0, i, 3] = v.pfMetric
            inputs_raw[0, i, 4] = v.delayFactor
        mask = inputs_raw[0, :, 0] != 0
        inputs.copy_(inputs_raw)
        inputs[0, mask, 1] = (inputs[0, mask, 1] - mean[0]) / std[0]
        inputs[0, mask, 3] = (inputs[0, mask, 3] - mean[1]) / std[1]
        inputs[0, mask, 4] = (inputs[0, mask, 4] - mean[2]) / std[2]
        pad_mask[0] = ~mask
        assert inputs_raw.shape == (1, FLOW_NUMS, 5)
        assert pad_mask.shape == (1, FLOW_NUMS)
        # print("Input tensor:", inputs)
        with torch.no_grad():
            # start_time = time.perf_counter()

            outputs = model(inputs, src_key_padding_mask=pad_mask)

            # end_time = time.perf_counter()
            # inference_time_ms = (end_time - start_time) * 1000

            assert outputs.shape == (1, 12, 4)
            # print("Raw model output size: ", outputs.shape)
            # print("Raw model output:", outputs)
            outputs = outputs.argmax(dim=-1).squeeze(0)  # Remove batch dimension (12, 4) and get argmax
            # print("Output size after argmax: ", outputs.shape)
            # print("Predicted symbols:", outputs)

            # with open(output_csv, "a", newline="") as f:
            #     writer = csv.writer(f)
            #     writer.writerow([
            #         f"{inference_time_ms:.4f}"
            #         ])
        
        out_vec = msgInterface.GetPy2CppVector()[0]
        out_vec.sym1 = int(outputs[0])
        out_vec.sym2 = int(outputs[1])
        out_vec.sym3 = int(outputs[2])
        out_vec.sym4 = int(outputs[3])
        out_vec.sym5 = int(outputs[4])
        out_vec.sym6 = int(outputs[5])
        out_vec.sym7 = int(outputs[6])
        out_vec.sym8 = int(outputs[7])
        out_vec.sym9 = int(outputs[8])
        out_vec.sym10 = int(outputs[9])
        out_vec.sym11 = int(outputs[10])
        out_vec.sym12 = int(outputs[11])
        
        msgInterface.PyRecvEnd()
        msgInterface.PySendEnd()

except Exception as e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print("Exception occurred: {}".format(e))
    print("Traceback:")
    traceback.print_tb(exc_traceback)
    exit(1)
else:
    pass

finally:
    print("Finally exiting...")
    del exp