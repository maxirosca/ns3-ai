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
output_csv = "/home/maximilianrosca/ns-3-dev/scheduler_time_only_model_inference.csv"

# Create file + header once
if not os.path.exists(output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "inference_time_ms"
        ])

parser = argparse.ArgumentParser()
parser.add_argument("--randomStream", type=int, required=True)
# parser.add_argument("--numerology", type=int, required=True)
# parser.add_argument("--scenario", type=str, required=True)
# parser.add_argument("--enableTransformer", type=bool, required=True)
args = parser.parse_args()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NrTransformerModel(d_model=16, nhead=8, num_layers=1)
model.load_state_dict(torch.load('nr_transformer_model_best_lr1e-05_dmodel16_nhead8_layers1_bs32.pth', map_location=torch.device('cpu'), weights_only=True), strict=True)
model = model.to(device)
model.eval()
input_tensor = torch.tensor([
    [1, 0, 1.89645732694928, -0.460449008821096, 0.224142628740084, -0.55675641025158, -0.213868534482615],
    [2, 1, 0.564031082547522, 1.00345553164235, 0.224142628740084, -0.553541058970402, 0.177493585282969],
    [2, 2, -1.10150172295467, -1.08260843851806, 0.224142628740084, 0.285191361102902, 0.177493585282969],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
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

assert mean.shape == (5,)
assert std.shape == (5,)

inputs = torch.zeros(1, FLOW_NUMS, 7, dtype=torch.float32, device=device)
inputs_raw = torch.zeros(1, FLOW_NUMS, 7, dtype=torch.float32, device=device)
pad_mask = torch.zeros((1, FLOW_NUMS), dtype=torch.bool, device=device)

gc.disable()

print("Starting nr_transformer_use_model.py...")
exp = Experiment("nr_transformer_demo", "../../../../", py_binding,
                handleFinish=True, useVector=True, vectorSize=FLOW_NUMS)
setting_map = {
    # "enableTransformer": args.enableTransformer,
    "randomStream": args.randomStream,
    # "numerology": args.numerology,
    # "scenario": args.scenario
    }
msgInterface = exp.run(setting_map, show_output=True)
# msgInterface = exp.run(show_output=True)
print("Experiment started...")

try:
    while True:
        # receive from C++ side
        msgInterface.PyRecvBegin()
        if msgInterface.PyGetFinished():
            break

        # send to C++ side
        msgInterface.PySendBegin()
        cpp_vec = msgInterface.GetCpp2PyVector()
        for i in range(FLOW_NUMS):
            v = cpp_vec[i]
            inputs_raw[0, i, 0] = v.rnti
            inputs_raw[0, i, 1] = v.resource_type
            inputs_raw[0, i, 2] = v.priority
            inputs_raw[0, i, 3] = v.packetDelayBudget
            inputs_raw[0, i, 4] = v.availableSymbols
            inputs_raw[0, i, 5] = v.queueSize
            inputs_raw[0, i, 6] = v.mcs
        mask = inputs_raw[0, :, 0] != 0
        inputs.copy_(inputs_raw)
        inputs[0, mask, 2:] = (inputs[0, mask, 2:] - mean) / std
        pad_mask[0] = ~mask
        assert inputs_raw.shape == (1, FLOW_NUMS, 7)
        assert pad_mask.shape == (1, FLOW_NUMS)
        # print("Input tensor:", inputs)
        with torch.no_grad():
            start_time = time.perf_counter()

            outputs = model(inputs, src_key_padding_mask=pad_mask)

            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000

            assert outputs.shape == (1, 12, 4)
            # print("Raw model output size: ", outputs.shape)
            # print("Raw model output:", outputs)
            outputs = outputs.argmax(dim=-1).squeeze(0)  # Remove batch dimension (12, 4) and get argmax
            # print("Output size after argmax: ", outputs.shape)
            # print("Predicted symbols:", outputs)

            with open(output_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    f"{inference_time_ms:.4f}"
                    ])
        
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


# import nr_transformer_interface_py_vec as py_binding
# from ns3ai_utils import Experiment
# import sys
# import traceback
# import torch
# from nr_transformer_model import NrTransformerModel
# import argparse
# import csv
# from datetime import datetime
# import os
# import time

# FLOW_NUMS = 5
# # output_csv = "scheduler_time_only_model_inference.csv"

# # # Create file + header once
# # if not os.path.exists(output_csv):
# #     with open(output_csv, "w", newline="") as f:
# #         writer = csv.writer(f)
# #         writer.writerow([
# #             # "timestamp",
# #             # "inference_id",
# #             # "inference time",
# #             # *[f"symbol_{i}_ue" for i in range(12)]
# #             "inference_time_ms"
# #         ])

# # parser = argparse.ArgumentParser()
# # parser.add_argument("--randomStream", type=int, required=True)
# # # parser.add_argument("--numerology", type=int, required=True)
# # parser.add_argument("--scenario", type=str, required=True)
# # # parser.add_argument("--enableTransformer", type=bool, required=True)
# # args = parser.parse_args()

# def normalize_inputs(x, mean, std):
#     x = x.clone()

#     # Mask padded rows (rnti == 0)
#     mask = x[:, 0] != 0

#     x[mask, 2:] = (x[mask, 2:] - mean) / std

#     return x

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")

# model = NrTransformerModel(d_model=16, nhead=1, num_layers=1)
# model.load_state_dict(torch.load('nr_transformer_model_best_lr1e-05_dmodel16_nhead1_layers1_bs16.pth', map_location=torch.device('cpu'), weights_only=True), strict=True)
# model = model.to(device)
# model.eval()
# input_tensor = torch.tensor([
#     [1, 0, 1.89645732694928, -0.460449008821096, 0.224142628740084, -0.55675641025158, -0.213868534482615],
#     [2, 1, 0.564031082547522, 1.00345553164235, 0.224142628740084, -0.553541058970402, 0.177493585282969],
#     [2, 2, -1.10150172295467, -1.08260843851806, 0.224142628740084, 0.285191361102902, 0.177493585282969],
#     [0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0]
# ], dtype=torch.float32).unsqueeze(0).to(device)
# pad_mask = (input_tensor[..., 0] == 0).to(device)

# # Warm-up inference
# with torch.no_grad():
#     for _ in range(10):
#         _ = model(input_tensor, src_key_padding_mask=pad_mask)

# # with torch.no_grad():
# #     assert not model.training, "Model is still in training mode!"
# #     inputs = inputs.unsqueeze(0)
# #     outputs1 = model(inputs).argmax(dim=-1)
# #     outputs2 = model(inputs).argmax(dim=-1)
# #     print("Δ:", torch.max(torch.abs(outputs1 - outputs2)))
# #     print("Pred1:", outputs1)
# #     print("Pred2:", outputs2)

#     #print("Pred:", pred.cpu().numpy())
# norm_params = torch.load("normalization_params.pt", map_location=torch.device('cpu'))
# mean = norm_params["mean"].to(device)
# std = norm_params["std"].to(device)

# assert mean.shape == (5,)
# assert std.shape == (5,)

# print("Starting nr_transformer_use_model.py...")
# exp = Experiment("nr_transformer_demo", "../../../../", py_binding,
#                 handleFinish=True, useVector=True, vectorSize=FLOW_NUMS)
# # setting_map = {
# #     # "enableTransformer": args.enableTransformer,
# #     "randomStream": args.randomStream,
# #     # "numerology": args.numerology,
# #     "scenario": args.scenario
# #     }
# # msgInterface = exp.run(setting_map, show_output=True)
# msgInterface = exp.run(show_output=True)
# print("Experiment started...")
# # inference_id = 0

# try:
#     while True:
#         # receive from C++ side
#         msgInterface.PyRecvBegin()
#         if msgInterface.PyGetFinished():
#             break

#         # send to C++ side
#         msgInterface.PySendBegin()
#         inputs = torch.tensor([
#             [
#                 msgInterface.GetCpp2PyVector()[i].rnti,
#                 msgInterface.GetCpp2PyVector()[i].resource_type,
#                 msgInterface.GetCpp2PyVector()[i].priority,
#                 msgInterface.GetCpp2PyVector()[i].packetDelayBudget,
#                 msgInterface.GetCpp2PyVector()[i].availableSymbols,
#                 msgInterface.GetCpp2PyVector()[i].queueSize,
#                 msgInterface.GetCpp2PyVector()[i].mcs
#             ] for i in range(len(msgInterface.GetCpp2PyVector()))
#             ], dtype=torch.float32).to(device)
#         inputs = normalize_inputs(inputs, mean, std)
#         # print("Input tensor:", inputs)
#         with torch.no_grad():
#             inputs = inputs.unsqueeze(0)  # Add batch dimension (1, 5 ,7)
#             pad_mask = (inputs[..., 0] == 0).to(device)  

#             assert inputs.shape == (1, FLOW_NUMS, 7)
#             assert pad_mask.shape == (1, FLOW_NUMS)

#             # start_time = time.perf_counter()

#             outputs = model(inputs, src_key_padding_mask=pad_mask)

#             # end_time = time.perf_counter()
#             # inference_time_ms = (end_time - start_time) * 1000

#             assert outputs.shape == (1, 12, 4)
#             # print("Raw model output size: ", outputs.shape)
#             # print("Raw model output:", outputs)
#             outputs = outputs.squeeze(0)  # Remove batch dimension (12, 4)
#             # print("Output size before softmax: ", outputs.shape)
#             # print("Output tensor before softmax:", outputs)
#             # outputs = outputs.softmax(dim=-1)
#             # print("Output size after softmax: ", outputs.shape)
#             # print("Output after softmax:", outputs)
#             outputs = outputs.argmax(dim=-1)
#             # print("Output size after argmax: ", outputs.shape)
#             # print("Predicted symbols:", outputs)
#             outputs = outputs.detach().cpu().numpy()

#             # with open(output_csv, "a", newline="") as f:
#             #     writer = csv.writer(f)
#             #     writer.writerow([
#             #         # datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             #         # inference_id,
#             #         f"{inference_time_ms:.4f}"
#             #         #*outputs.tolist()
#             #         ])
#             # inference_id += 1
        
#         msgInterface.GetPy2CppVector()[0].sym1 = int(outputs[0])
#         msgInterface.GetPy2CppVector()[0].sym2 = int(outputs[1])
#         msgInterface.GetPy2CppVector()[0].sym3 = int(outputs[2])
#         msgInterface.GetPy2CppVector()[0].sym4 = int(outputs[3])
#         msgInterface.GetPy2CppVector()[0].sym5 = int(outputs[4])
#         msgInterface.GetPy2CppVector()[0].sym6 = int(outputs[5])
#         msgInterface.GetPy2CppVector()[0].sym7 = int(outputs[6])
#         msgInterface.GetPy2CppVector()[0].sym8 = int(outputs[7])
#         msgInterface.GetPy2CppVector()[0].sym9 = int(outputs[8])
#         msgInterface.GetPy2CppVector()[0].sym10 = int(outputs[9])
#         msgInterface.GetPy2CppVector()[0].sym11 = int(outputs[10])
#         msgInterface.GetPy2CppVector()[0].sym12 = int(outputs[11])
        
#         msgInterface.PyRecvEnd()
#         msgInterface.PySendEnd()

# except Exception as e:
#     exc_type, exc_value, exc_traceback = sys.exc_info()
#     print("Exception occurred: {}".format(e))
#     print("Traceback:")
#     traceback.print_tb(exc_traceback)
#     exit(1)
# else:
#     pass

# finally:
#     print("Finally exiting...")
#     del exp