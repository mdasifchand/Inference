# Inference

- Resnet50 architecture is used, the main reason is, it democratized the usage of skip connections with residuals which helped DL networks to go deeper and at the same time improving efficiency. ResNet50 even till date is used as a feature extractor. Ex: Yolo etc.

- For more details read the research paper. ResNet50 here performs classifications in real-time with minimal latency and maximum throughput. The program is designed to run on AI enabled embedded automotive hardware. 

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




