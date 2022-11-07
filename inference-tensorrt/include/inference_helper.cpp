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

std::string trt::InferenceEngine::serializeEngine(const Settings& settings)
{
    std::string engineName = "dynamic_restnet50-tuned.engine";
    std::vector<std::string> gpuUUIDs;
    getGPUUUIDS(gpuUUIDs);
    if (static_cast<size_t>(settings.deviceIndex) >= gpuUUIDs.size())
    {
        throw std::runtime_error("Error, out of range device index");
    }
    engineName += "." + gpuUUIDs[settings.deviceIndex];
    if (settings.FP16)
    {
        engineName += ".fp16";
    }
    else
    {
        engineName += ".fp32";
    }
    engineName += "." + std::to_string(settings.maxBatchSize) + ".";
}

bool trt::InferenceEngine::buildNetwork(std::string OnnxModelPath)
{

    (void) OnnxModelPath;
    return true;
}

bool trt::InferenceEngine::loadNetwork()
{
    return true;
}

bool trt::InferenceEngine::runInference(const std::vector<cv::Mat>& inputImage)
{
    (void) inputImage;
    return true;
}

void trt::InferenceEngine::getGPUUUIDS(std::vector<std::string>& gpuUUIDS)
{
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    for (int i = 0; i < numGPUs; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        char uuid[33];
        for (int i = 0; i < 16; i++)
        {
            sprintf(&uuid[i * 2], "%02x", (unsigned char) prop.uuid.bytes[i]);
        }
        gpuUUIDS.push_back(std::string(uuid));
    }
}

bool trt::InferenceEngine::getFileStatus(const std::filesystem::path& path)
{
    return std::filesystem::exists(path);
}
