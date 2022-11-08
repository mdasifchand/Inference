#include "inference_helper.hpp"

/*TODO:
- Create an ONNX model
- Finetune it
- Parse and use it
*/
int main()
{
    trt::Settings settings;
    settings.optBatchSize = {2};
    trt::InferenceEngine engine(settings);
    const std::string OnnxModelPath = "onnx/dynamic_restnet50-tuned.onnx";
    if (!engine.buildNetwork(OnnxModelPath))
    {
        throw std::runtime_error("Engine file cannot be generated");
    }

    if (!engine.loadNetwork())
    {
        throw std::runtime_error("Unable to load TensorRT engine file");
    }
    

    return 0;
}