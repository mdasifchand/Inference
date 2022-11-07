#include "inference_helper.hpp"

void trt::Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING)
    {
        std::cout << "The warning has been set to" << msg << std::endl;
    }
}

trt::InferenceEngine::InferenceEngine(const Settings& settings)
    : settings_(settings)
{
}

bool trt::InferenceEngine::buildNetwork(std::string OnnxModelPath) {}

bool trt::InferenceEngine::loadNetwork() {}

bool trt::InferenceEngine::runInference(const std::vector<cv::Mat>& inputImage) {}

std::string trt::InferenceEngine::serializeEngine(const Settings& setting) {}

void trt::InferenceEngine::getGPUUUIDS(std::vector<std::string>& gpuUUIDS) {}

bool trt::InferenceEngine::getFileStatus(const std::string& path) {}
