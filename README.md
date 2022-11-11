# Inference

- Resnet50 architecture is used, the main reason is, it democratized the usage of skip connections with residuals which helped DL networks to go deeper and at the same time improving efficiency. ResNet50 even till date is used as a feature extractor. Ex: Yolo etc.

- For more details read the research paper (included inside the repo). ResNet50 here performs classifications in real-time with minimal latency and maximum throughput. The program is designed to run on AI enabled embedded automotive hardware. 

# TODO:
- Output is represented in the form of numbers
- Create a dictionary and map numbers to classes (file located at Inference/models/resnet50/classes.txt)
- Create an example using apache-tvm
- Use polygraphy to compare runtime latencies between tvm and trt


# Dependencies
- Pytorch
- ONNX
- OpenCV
- TBB (for intel processors using std::execution)
- CUDA runtime
- TensorRT

# Building Project

- place engine file at `execution-engines` folder
- create a build directory and perform `cmake ..`
- from the root of the project run `./build/inference-tensorrt/inference-tensorrt

# Pipeline
- Find a problem (here: Classification)
- Look for relevant ML model (Here NN)
- Train Neural Network model 
- Quantize or prune model (if needed)
- Generate a dynamic batch size based ONNX model with optimizations (if needed like constant folding etc)
- Generate an optimized engine file 
- Build Network from ONNX
- create CudaEngine, Context and Stream
- Enqueue CudaStream
- Make inference

# I don't understand what's happening here: 

- Most of heavy computational models are not designed to run on (resource constrained) embedded systems, especially in case of ML models which use Deep Learning in the background. As a result they cannot be used in real world scenarios where achieving soft/hard deadlines are quite important. 

- Python by itself is an interpreter language and it cannot generate machine level code, therefore we need more of those C++ wrappers to bind ML models to specific processor achitecture and make them run as faster as they could and at the same time utilising resources effectively. [interesting talk about the same](https://www.youtube.com/watch?v=3SypMvnQT_s&ab_channel=TeslaOwnersOnline)

- Vendors like Nvidia also provide DeepLearning specific registers like DLA ex:[Drive_Platform](https://www.nvidia.com/de-de/self-driving-cars/drive-platform/hardware/) , I have also known the infineon [Aurix TC4](https://www.youtube.com/watch?v=vy964dkk67I&ab_channel=Synopsys) series to have PPU's







