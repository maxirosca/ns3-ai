import nr_transformer_interface_py_vec as py_binding
from ns3ai_utils import Experiment
import sys
import traceback

UE_NUMS = 2
ALPHA = 1
print("Starting nr_transformer_interface.py...")
exp = Experiment("nr_transformer_interface_vec", "../../../../", py_binding,
                 handleFinish=True, useVector=True, vectorSize=UE_NUMS)
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
        for i in range(len(msgInterface.GetCpp2PyVector())):
            msgInterface.GetPy2CppVector()[i].weight = (100 - msgInterface.GetCpp2PyVector()[i].priority) \
                * pow(msgInterface.GetCpp2PyVector()[i].potThroughput, ALPHA) \
                / max(1e-9, msgInterface.GetCpp2PyVector()[i].avgThroughput) \
                * msgInterface.GetCpp2PyVector()[i].delayBudget
        # send weights to c++
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
