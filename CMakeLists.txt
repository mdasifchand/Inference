cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 14)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
add_subdirectory(inference-tensorrt)
add_subdirectory(inference-apache-tvm)

