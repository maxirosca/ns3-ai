import nr_transformer_interface_py_vec as py_binding
from ns3ai_utils import Experiment
import sys
import traceback
import torch
from nr_transformer_model import NrTransformerModel

# import argparse

FLOW_NUMS = 5

# parser = argparse.ArgumentParser()
# parser.add_argument("--randomStream", type=int, required=True)
# parser.add_argument("--numerology", type=int, required=True)
# parser.add_argument("--scenario", type=str, required=True)
# args = parser.parse_args()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NrTransformerModel(d_model=64, nhead=8, num_layers=2, num_ues=4)
model.load_state_dict(torch.load('nr_transformer_model_best_lr0.0005_dmodel64_nhead8_layers2_bs32.pth', map_location=torch.device('cpu'), weights_only=True))
model = model.to(device)
model.eval()

print("Starting nr_transformer_use_model.py...")
exp = Experiment("nr_transformer_demo", "../../../../", py_binding,
                handleFinish=True, useVector=True, vectorSize=FLOW_NUMS)
# setting_map = {
#     "randomStream": args.randomStream,
#     "numerology": args.numerology,
#     "scenario": args.scenario
#     }
# msgInterface = exp.run(setting_map, show_output=True)device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
msgInterface = exp.run(show_output=True)
print("Experiment started...")

try:
    while True:
        # receive from C++ side
        msgInterface.PyRecvBegin()
        if msgInterface.PyGetFinished():
            break

        # send to C++ side
        msgInterface.PySendBegin()
        inputs = torch.tensor([
            [
                msgInterface.GetCpp2PyVector()[i].rnti,
                msgInterface.GetCpp2PyVector()[i].resource_type,
                msgInterface.GetCpp2PyVector()[i].priority,
                msgInterface.GetCpp2PyVector()[i].packetDelayBudget,
                msgInterface.GetCpp2PyVector()[i].queueSize,
                msgInterface.GetCpp2PyVector()[i].availableSymbols,
                msgInterface.GetCpp2PyVector()[i].mcs
            ] for i in range(len(msgInterface.GetCpp2PyVector()))
            ], dtype=torch.float32).to(device)
        # print("Input tensor:", inputs)
        with torch.no_grad():
            inputs = inputs.unsqueeze(0)  # Add batch dimension (1, 5 ,7)
            outputs = model(inputs)
            outputs = outputs.squeeze(0)  # Remove batch dimension (12)
            # print("Output size: ", outputs.shape)
            outputs = outputs.softmax(dim=-1)
            # print("Output after softmax:", outputs)
            outputs = outputs.argmax(dim=-1)
            # print("Predicted symbols:", outputs)
            outputs = outputs.detach().cpu().numpy()
            
        msgInterface.GetPy2CppVector()[0].sym1 = int(outputs[0])
        msgInterface.GetPy2CppVector()[0].sym2 = int(outputs[1])
        msgInterface.GetPy2CppVector()[0].sym3 = int(outputs[2])
        msgInterface.GetPy2CppVector()[0].sym4 = int(outputs[3])
        msgInterface.GetPy2CppVector()[0].sym5 = int(outputs[4])
        msgInterface.GetPy2CppVector()[0].sym6 = int(outputs[5])
        msgInterface.GetPy2CppVector()[0].sym7 = int(outputs[6])
        msgInterface.GetPy2CppVector()[0].sym8 = int(outputs[7])
        msgInterface.GetPy2CppVector()[0].sym9 = int(outputs[8])
        msgInterface.GetPy2CppVector()[0].sym10 = int(outputs[9])
        msgInterface.GetPy2CppVector()[0].sym11 = int(outputs[10])
        msgInterface.GetPy2CppVector()[0].sym12 = int(outputs[11])
        
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