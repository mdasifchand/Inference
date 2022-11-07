cmake_minimum_required(VERSION 3.10)
project(inference-tensorrt)

#find_package(OpenCV)
find_package(OpenCV REQUIRED)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(inference_helper SHARED include/inference_helper.cpp)
target_include_directories(inference_helper PUBLIC ${CMAKE_CURRENT_PROJECT_DIR}/include /workspace/tensorrt/samples/common ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inference_helper PUBLIC  ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} -pthread ${Opencv_LIBS})


add_executable(inference-tensorrt src/main.cpp)
target_include_directories(inference-tensorrt PUBLIC ${CMAKE_CURRENT_PROJECT_DIR}/include /workspace/tensorrt/samples/common ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inference-tensorrt inference_helper)

