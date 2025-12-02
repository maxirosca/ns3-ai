#include "ns3/nr-transformer-interface.h"

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
        .def_readwrite("rnti", &FeaturesStruct::rnti)
        .def_readwrite("resource_type", &FeaturesStruct::resource_type)
        .def_readwrite("priority", &FeaturesStruct::priority)
        .def_readwrite("packetDelayBudget", &FeaturesStruct::packetDelayBudget)
        .def_readwrite("queueSize", &FeaturesStruct::queueSize)
        .def_readwrite("availableSymbols", &FeaturesStruct::availableSymbols)
        .def_readwrite("mcs", &FeaturesStruct::mcs);

    py::class_<ActStruct>(m, "PyActStruct")
        .def(py::init<>())
        .def_readwrite("sym1", &ActStruct::sym1)
        .def_readwrite("sym2", &ActStruct::sym2)
        .def_readwrite("sym3", &ActStruct::sym3)
        .def_readwrite("sym4", &ActStruct::sym4)
        .def_readwrite("sym5", &ActStruct::sym5)
        .def_readwrite("sym6", &ActStruct::sym6)
        .def_readwrite("sym7", &ActStruct::sym7)
        .def_readwrite("sym8", &ActStruct::sym8)
        .def_readwrite("sym9", &ActStruct::sym9)
        .def_readwrite("sym10", &ActStruct::sym10)
        .def_readwrite("sym11", &ActStruct::sym11)
        .def_readwrite("sym12", &ActStruct::sym12);

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
