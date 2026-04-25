cmake_minimum_required(VERSION 3.18)
project(cuda_dl_ops VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

enable_testing()

add_subdirectory(tests)
add_subdirectory(scripts)

# Benchmark executable
add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark cuda_ops)
target_include_directories(benchmark PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Shared library for Python binding
add_library(cuda_ops_shared SHARED
    src/cuda_ops_export.cu
    src/cross_entropy.cu
    src/sgd_update.cu
    src/flatten.cu
)
target_include_directories(cuda_ops_shared PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(cuda_ops_shared CUDA::cudart)
set_target_properties(cuda_ops_shared PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    POSITION_INDEPENDENT_CODE ON
)
