#include "nr-transformer-interface.h"

#include <ns3/ai-module.h>

#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::Cpp2PyMsgVector);
PYBIND11_MAKE_OPAQUE(ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::Py2CppMsgVector);

PYBIND11_MODULE(nr_transformer_interface_py_vec, m)
{
    py::class_<FeaturesStruct>(m, "PyEnvStruct")
        .def(py::init<>())
        .def_readwrite("qci", &FeaturesStruct::qci)
        .def_readwrite("priority", &FeaturesStruct::priority)
        .def_readwrite("holDelay", &FeaturesStruct::holDelay)
        .def_readwrite("delayBudget", &FeaturesStruct::delayBudget)
        .def_readwrite("avgThroughput", &FeaturesStruct::avgThroughput)
        .def_readwrite("potThroughput", &FeaturesStruct::potThroughput);

    py::class_<ActStruct>(m, "PyActStruct").def(py::init<>()).def_readwrite("weight", &ActStruct::weight);

    py::class_<ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::Cpp2PyMsgVector>(m, "PyEnvVector")
        .def(
            "resize",
            static_cast<void (ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::Cpp2PyMsgVector::*)(
                ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::Cpp2PyMsgVector::size_type)>(
                &ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::Cpp2PyMsgVector::resize))
        .def("__len__", &ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::Cpp2PyMsgVector::size)
        .def(
            "__getitem__",
            [](ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::Cpp2PyMsgVector& vec,
               uint32_t i) -> FeaturesStruct& {
                if (i >= vec.size())
                {
                    std::cerr << "Invalid index " << i << " for vector, whose size is "
                              << vec.size() << std::endl;
                    exit(1);
                }
                return vec.at(i);
            },
            py::return_value_policy::reference);

    py::class_<ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::Py2CppMsgVector>(m, "PyActVector")
        .def(
            "resize",
            static_cast<void (ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::Py2CppMsgVector::*)(
                ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::Py2CppMsgVector::size_type)>(
                &ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::Py2CppMsgVector::resize))
        .def("__len__", &ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::Py2CppMsgVector::size)
        .def(
            "__getitem__",
            [](ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::Py2CppMsgVector& vec,
               uint32_t i) -> ActStruct& {
                if (i >= vec.size())
                {
                    std::cerr << "Invalid index " << i << " for vector, whose size is "
                              << vec.size() << std::endl;
                    exit(1);
                }
                return vec.at(i);
            },
            py::return_value_policy::reference);

    py::class_<ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>>(m, "Ns3AiMsgInterfaceImpl")
        .def(py::init<bool,
                      bool,
                      bool,
                      uint32_t,
                      const char*,
                      const char*,
                      const char*,
                      const char*>())
        .def("PyRecvBegin", &ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::PyRecvBegin)
        .def("PyRecvEnd", &ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::PyRecvEnd)
        .def("PySendBegin", &ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::PySendBegin)
        .def("PySendEnd", &ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::PySendEnd)
        .def("PyGetFinished", &ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::PyGetFinished)
        .def("GetCpp2PyVector",
             &ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::GetCpp2PyVector,
             py::return_value_policy::reference)
        .def("GetPy2CppVector",
             &ns3::Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>::GetPy2CppVector,
             py::return_value_policy::reference);
}
