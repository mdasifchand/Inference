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
    // ignored optimal batch size for now
    engineName += "." + std::to_string(settings.maxWorkSpaceSize);
    return engineName;
}

bool trt::InferenceEngine::buildNetwork(std::string OnnxModelPath)
{
    engineName_ = serializeEngine(settings_);
    std::cout << "Searching for engine file with name :" << engineName_ << std::endl;
    if (getFileStatus(engineName_))
    {
        std::cout << "Engine exists" << std::endl;
        return true;
    }
    std::cout << "Engine not found" << std::endl;
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
    if (!builder)
    {
        std::cout << "Builder is throwing an error" << std::endl;
        return false;
    }
    builder->setMaxBatchSize(settings_.maxBatchSize);
    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
    if (!parser)
    {
        std::cout << "onnx model cannot be parsed" << std::endl;
        return false;
    }
    std::ifstream file(OnnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
    {
        throw std::runtime_error("Unable to read engine file");
    }
    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed)
    {
        return false;
    }
    const auto input = network->getInput(0);
    const auto output = network->getOutput(0);
    const auto outDims = output->getDimensions();
    const auto inputName = input->getName();
    const auto inputDims = input->getDimensions();
    int32_t inputC = inputDims.d[1];
    int32_t inputH = inputDims.d[2];
    int32_t inputW = inputDims.d[3];
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }
    IOptimizationProfile* defaultProfile = builder->createOptimizationProfile();
    defaultProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
    defaultProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(1, inputC, inputH, inputW));
    defaultProfile->setDimensions(
        inputName, OptProfileSelector::kOPT, Dims4(settings_.maxBatchSize, inputC, inputH, inputW));
    config->addOptimizationProfile(defaultProfile);
    for (auto i = 1; i < settings_.maxBatchSize; i++)
    {
        IOptimizationProfile* profile = builder->createOptimizationProfile();
        profile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
        profile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(i, inputC, inputH, inputW));
        profile->setDimensions(
            inputName, OptProfileSelector::kMAX, Dims4(settings_.maxBatchSize, inputC, inputH, inputW));
        config->addOptimizationProfile(profile);
    }
    config->setMaxWorkspaceSize(settings_.maxWorkSpaceSize);
    if (settings_.FP16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    auto profileStream = samplesCommon::makeCudaStream();
    config->setProfileStream(*profileStream);
    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }
    std::ofstream outFile(engineName_, std::ios::binary);
    std::cout << "Successfully saved serialized engine file" << engineName_ << std::endl;
    return true;
}

bool trt::InferenceEngine::loadNetwork()
{
    std::ifstream file(engineName_, std::ios::binary | std::ios::ate);
    std::streamsize ssize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(ssize);
    file.read(buffer.data(), ssize);
    std::unique_ptr<IRuntime> runtime{createInferRuntime(logger_)};
    auto rc = cudaSetDevice(settings_.deviceIndex);
    if (rc != 0)
    {
        int numGPUs = -1;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(settings_.deviceIndex)
            + ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    auto cudaRetCode = cudaStreamCreate(&cudaStream_);
    return true;
}

bool trt::InferenceEngine::runInference(
    const std::vector<cv::Mat>& inputImage, std::vector<std::vector<float>>& featureVectors)
{
    auto dims = engine_->getBindingDimensions(0);
    auto outputLength = engine_->getBindingDimensions(1);

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

trt::InferenceEngine::~InferenceEngine()
{
    if (cudaStream_)
    {
        cudaStreamDestroy(cudaStream_);
    }
}