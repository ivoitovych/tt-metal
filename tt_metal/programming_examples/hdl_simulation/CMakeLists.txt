set(HDL_SIMULATION_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/basic_simulation/basic_simulation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/host_comm_simulation/host_comm_simulation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_communication/simple_communication_example.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simplest_communication/simplest_communication_example.cpp
    #   ${CMAKE_CURRENT_SOURCE_DIR}/mini_communication/mini_communication.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/minimal_communication/minimal_communication.cpp
)

CREATE_PGM_EXAMPLES_EXE("${HDL_SIMULATION_SRCS}" "hdl_simulation")

add_custom_target(hdl_simulation_examples DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
