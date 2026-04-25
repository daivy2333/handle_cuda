add_custom_target(run_tests
    COMMAND ${CMAKE_BINARY_DIR}/bin/test_matmul
    COMMAND ${CMAKE_BINARY_DIR}/bin/test_relu
    COMMAND ${CMAKE_BINARY_DIR}/bin/test_bias_add
    COMMAND ${CMAKE_BINARY_DIR}/bin/test_softmax
    COMMAND ${CMAKE_BINARY_DIR}/bin/test_conv2d
    COMMAND ${CMAKE_BINARY_DIR}/bin/test_maxpool2d
    COMMENT "Running all CUDA operator tests"
)
