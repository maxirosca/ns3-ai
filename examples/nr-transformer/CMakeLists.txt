build_lib_example(
        NAME nr_transformer_interface_vec
        SOURCE_FILES
                nr-transformer-interface.cc
                maxi-scheduler-test-scenario.cc
        LIBRARIES_TO_LINK
                ${libai} ${libflow-monitor} 
                ${libnr} 
)

pybind11_add_module(nr_transformer_interface_py_vec nr-transformer-interface_py.cc)
set_target_properties(nr_transformer_interface_py_vec PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
target_link_library(nr_transformer_interface_py_vec PRIVATE ${libai})

add_dependencies(nr_transformer_interface_vec nr_transformer_interface_py_vec)