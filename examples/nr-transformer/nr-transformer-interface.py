import nr_transformer_interface_py_vec as py_binding
from ns3ai_utils import Experiment
import sys
import traceback

UE_NUMS = 2

exp = Experiment("nr_transformer_interface_py_vec", "../../../../../", py_binding,
                 handleFinish=True, useVector=True, vectorSize=UE_NUMS)
msgInterface = exp.run(show_output=True)

try:
    while True:
        # receive from C++ side
        msgInterface.PyRecvBegin()
        if msgInterface.PyGetFinished():
            break

        # send to C++ side
        msgInterface.PySendBegin()
        # ToDO:
        # calculate the weights
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
