set(CUDA_OPS_SOURCES
    ${CMAKE_SOURCE_DIR}/src/matmul.cu
    ${CMAKE_SOURCE_DIR}/src/matmul_fp16.cu
    ${CMAKE_SOURCE_DIR}/src/matmul_cublas.cu
    ${CMAKE_SOURCE_DIR}/src/half_utils.cu
    ${CMAKE_SOURCE_DIR}/src/loss_scaling.cu
    ${CMAKE_SOURCE_DIR}/src/relu.cu
    ${CMAKE_SOURCE_DIR}/src/bias_add.cu
    ${CMAKE_SOURCE_DIR}/src/softmax.cu
    ${CMAKE_SOURCE_DIR}/src/conv2d.cu
    ${CMAKE_SOURCE_DIR}/src/conv2d_cublas.cu
    ${CMAKE_SOURCE_DIR}/src/conv2d_winograd.cu
    ${CMAKE_SOURCE_DIR}/src/conv2d_winograd_f6.cu
    ${CMAKE_SOURCE_DIR}/src/conv2d_fused.cu
    ${CMAKE_SOURCE_DIR}/src/maxpool2d.cu
    ${CMAKE_SOURCE_DIR}/src/sigmoid.cu
    ${CMAKE_SOURCE_DIR}/src/tanh.cu
    ${CMAKE_SOURCE_DIR}/src/dropout.cu
    ${CMAKE_SOURCE_DIR}/src/cross_entropy.cu
    ${CMAKE_SOURCE_DIR}/src/sgd_update.cu
    ${CMAKE_SOURCE_DIR}/src/flatten.cu
)

add_library(cuda_ops STATIC ${CUDA_OPS_SOURCES})
target_include_directories(cuda_ops PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(cuda_ops PUBLIC CUDA::cudart CUDA::cublas)

add_executable(test_matmul test_matmul.cpp)
target_link_libraries(test_matmul cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_matmul PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_relu test_relu.cpp)
target_link_libraries(test_relu cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_relu PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_bias_add test_bias_add.cpp)
target_link_libraries(test_bias_add cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_bias_add PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_softmax test_softmax.cpp)
target_link_libraries(test_softmax cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_softmax PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_conv2d test_conv2d.cpp)
target_link_libraries(test_conv2d cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_conv2d PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_maxpool2d test_maxpool2d.cpp)
target_link_libraries(test_maxpool2d cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_maxpool2d PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_sigmoid test_sigmoid.cpp)
target_link_libraries(test_sigmoid cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_sigmoid PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_tanh test_tanh.cpp)
target_link_libraries(test_tanh cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_tanh PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_dropout test_dropout.cpp)
target_link_libraries(test_dropout cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_dropout PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_cross_entropy test_cross_entropy.cpp)
target_link_libraries(test_cross_entropy cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_cross_entropy PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_sgd_update test_sgd_update.cpp)
target_link_libraries(test_sgd_update cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_sgd_update PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_flatten test_flatten.cpp)
target_link_libraries(test_flatten cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_flatten PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_edge_cases test_edge_cases.cpp)
target_link_libraries(test_edge_cases cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_edge_cases PRIVATE ${CMAKE_SOURCE_DIR}/include)

include(GoogleTest)
gtest_discover_tests(test_matmul)
gtest_discover_tests(test_relu)
gtest_discover_tests(test_bias_add)
gtest_discover_tests(test_softmax)
gtest_discover_tests(test_conv2d)
gtest_discover_tests(test_maxpool2d)
gtest_discover_tests(test_sigmoid)
gtest_discover_tests(test_tanh)
gtest_discover_tests(test_dropout)
gtest_discover_tests(test_cross_entropy)
gtest_discover_tests(test_sgd_update)
gtest_discover_tests(test_flatten)
gtest_discover_tests(test_edge_cases)

add_executable(test_conv2d_winograd test_conv2d_winograd.cpp)
target_link_libraries(test_conv2d_winograd cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_conv2d_winograd PRIVATE ${CMAKE_SOURCE_DIR}/include)
gtest_discover_tests(test_conv2d_winograd)

add_executable(test_conv2d_winograd_f6 test_conv2d_winograd_f6.cpp)
target_link_libraries(test_conv2d_winograd_f6 cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_conv2d_winograd_f6 PRIVATE ${CMAKE_SOURCE_DIR}/include)
gtest_discover_tests(test_conv2d_winograd_f6)

add_executable(test_fp16_tensor_core test_fp16_tensor_core.cpp)
target_link_libraries(test_fp16_tensor_core cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_fp16_tensor_core PRIVATE ${CMAKE_SOURCE_DIR}/include)
gtest_discover_tests(test_fp16_tensor_core)

add_executable(test_tensor_core_optimized test_tensor_core_optimized.cpp)
target_link_libraries(test_tensor_core_optimized cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_tensor_core_optimized PRIVATE ${CMAKE_SOURCE_DIR}/include)
gtest_discover_tests(test_tensor_core_optimized)

add_executable(test_conv2d_fused test_conv2d_fused.cpp)
target_link_libraries(test_conv2d_fused cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_conv2d_fused PRIVATE ${CMAKE_SOURCE_DIR}/include)
gtest_discover_tests(test_conv2d_fused)

add_executable(test_matmul_cublas test_matmul_cublas.cpp)
target_link_libraries(test_matmul_cublas cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_matmul_cublas PRIVATE ${CMAKE_SOURCE_DIR}/include)
gtest_discover_tests(test_matmul_cublas)

add_executable(test_fp16_mixed_precision test_fp16_mixed_precision.cpp)
target_link_libraries(test_fp16_mixed_precision cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_fp16_mixed_precision PRIVATE ${CMAKE_SOURCE_DIR}/include)
gtest_discover_tests(test_fp16_mixed_precision)

add_executable(test_conv2d_cublas test_conv2d_cublas.cpp)
target_link_libraries(test_conv2d_cublas cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_conv2d_cublas PRIVATE ${CMAKE_SOURCE_DIR}/include)
gtest_discover_tests(test_conv2d_cublas)
