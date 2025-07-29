import nr_transformer_interface_py_vec as py_binding
from ns3ai_utils import Experiment
import sys
import traceback
import torch
from nr_transformer_model import NrTransformerModel
import argparse

UE_NUMS = 2
# ALPHA = 1

parser = argparse.ArgumentParser()
parser.add_argument("--randomStream", type=int, required=True)
parser.add_argument("--numerology", type=int, required=True)
parser.add_argument("--scenario", type=str, required=True)
args = parser.parse_args()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NrTransformerModel(input_dim=6)
model.load_state_dict(torch.load('nr_transformer_model.pth', weights_only=True))
model = model.to(device)
model.eval()

print("Starting nr_transformer_use_model.py...")
exp = Experiment("nr_transformer_demo", "../../../../", py_binding,
                handleFinish=True, useVector=True, vectorSize=UE_NUMS)
setting_map = {
    "randomStream": args.randomStream,
    "numerology": args.numerology,
    "scenario": args.scenario
    }
msgInterface = exp.run(setting_map, show_output=True)
print("Experiment started...")

try:
    while True:
        # receive from C++ side
        msgInterface.PyRecvBegin()
        if msgInterface.PyGetFinished():
            break

        # send to C++ side
        msgInterface.PySendBegin()
        # for i in range(len(msgInterface.GetCpp2PyVector())):
        # msgInterface.GetPy2CppVector()[i].weight = (100 - msgInterface.GetCpp2PyVector()[i].priority) \
        #     * pow(msgInterface.GetCpp2PyVector()[i].potThroughput, ALPHA) \
        #     / max(1e-9, msgInterface.GetCpp2PyVector()[i].avgThroughput) \
        #     * msgInterface.GetCpp2PyVector()[i].delayBudget
        # send weights to c++
        inputs = torch.tensor([
            [
                msgInterface.GetCpp2PyVector()[i].qci,
                msgInterface.GetCpp2PyVector()[i].priority,
                msgInterface.GetCpp2PyVector()[i].holDelay,
                msgInterface.GetCpp2PyVector()[i].delayBudget,
                msgInterface.GetCpp2PyVector()[i].avgThroughput,
                msgInterface.GetCpp2PyVector()[i].potThroughput
            ] for i in range(len(msgInterface.GetCpp2PyVector()))
            ], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(inputs)
            outputs = outputs.detach().cpu().numpy()
            
        for i in range(len(msgInterface.GetCpp2PyVector())):
            msgInterface.GetPy2CppVector()[i].weight = outputs[0][i]

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