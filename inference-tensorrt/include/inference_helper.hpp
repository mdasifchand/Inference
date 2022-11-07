#ifndef INFERENCE_HELPER_HPP_
#define INFERENCE_HELPER_HPP_
#include "NvInfer.h"
#include "buffers.h"

#include <opencv2/opencv.hpp>
#include <vector>

namespace trt
{

struct Settings
{
    bool FP16 = false;
    std::vector<int32_t> batchSize;
    int32_t maxBatchSize = 8;
    size_t maxWorkSpaceSize = 4000000000;
    int deviceIndex = 0;
};

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override;
};

class InferenceEngine
{
public:
    InferenceEngine(const Settings& settings);
    ~InferenceEngine();
    bool buildNetwork(std::string OnnxModelPath);
    bool loadNetwork();
    bool runInference(const std::vector<cv::Mat>& inputImage);

private:
    std::string serializeEngine(const Settings& setting);
    void getGPUUUIDS(std::vector<std::string>& gpuUUIDS);
    bool getFileStatus(const std::string& path);

    std::unique_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> context_ = nullptr;
    const Settings settings_;
    Logger logger_;
    samplesCommon::ManagedBuffer inputBuffer_;
    samplesCommon::ManagedBuffer outputBuffer_;
    size_t prevBatchSize = 0;
    std::string engineName_;
    cudaStream_t cudaStream_ = nullptr;
}

} // namespace trt

#endif