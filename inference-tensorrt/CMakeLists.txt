cmake_minimum_required(VERSION 3.10)
project(inference-tensorrt)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 14)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

#find_package(OpenCV)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)

add_executable(inference-tensorrt src/main.cpp)
target_include_directories(inference-tensorrt PUBLIC /workspace/tensorrt/samples/common ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ) #${opencv_INCLUDE_DIRS}
target_link_libraries(inference-tensorrt PUBLIC  ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ) #${opencv_LIBRARIES}

