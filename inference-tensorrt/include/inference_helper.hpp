#ifndef INFERENCE_HELPER_HPP_
#define INFERENCE_HELPER_HPP_

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "buffers.h"

#include <filesystem>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>

namespace trt
{
struct Settings
{
    bool FP16 = false;
    std::vector<int32_t> optBatchSize{3};
    int32_t maxBatchSize = 5;
    size_t maxWorkSpaceSize = 3500000000;
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
    bool buildNetwork(std::string OnnxModelPath, bool createEngine);
    bool loadNetwork();
    bool runInference(const std::vector<cv::Mat>& inputImage, std::vector<std::vector<float>>& featureVectors);
    std::vector<int> softMax(const int& batchSize, const std::vector<std::vector<float>>& outputVector);
    void setEngineName(const std::string& engineName);

private:
    std::string serializeEngine(const Settings& settings);
    void getGPUUUIDS(std::vector<std::string>& gpuUUIDS);
    bool getFileStatus(const std::filesystem::path& path);
    bool fullyConnectedLayerClasses(const std::filesystem::path& path);
    std::unique_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> context_ = nullptr;
    const Settings settings_;
    Logger logger_;
    samplesCommon::ManagedBuffer inputBuffer_;
    samplesCommon::ManagedBuffer outputBuffer_;
    size_t prevBatchSize = 0;
    std::string engineName_;
    cudaStream_t cudaStream_ = nullptr;
};

} // namespace trt

#endif