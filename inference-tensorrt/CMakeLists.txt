cmake_minimum_required(VERSION 3.10)
project(inference-tensorrt CUDA CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(OpenCV)
message(STATUS "opencv include dir: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "opencv lib dir: ${Opencv_LIBS}")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(inference_helper SHARED include/inference_helper.cpp include/inference_helper.hpp)
#target_compile_options(inference_helper PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
target_include_directories(inference_helper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include /workspace/tensorrt/samples/common ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inference_helper PUBLIC  ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} -pthread ${OpenCV_LIBS}  )


add_executable(inference-tensorrt src/main.cpp)
target_compile_options(inference-tensorrt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
target_include_directories(inference-tensorrt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inference-tensorrt inference_helper nvonnxparser ${OpenCV_LIBS} )

