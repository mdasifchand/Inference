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
    engine.buildNetwork(OnnxModelPath);

    return 0;
}