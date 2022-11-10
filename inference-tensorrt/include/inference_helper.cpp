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

bool trt::InferenceEngine::buildNetwork(std::string OnnxModelPath, bool createEngine)
{
    if (!createEngine)
    {
        if (engineName_.empty())
        {
            throw std::runtime_error("please provide an engine path");
        }
        else
        {
            std::cout << "Engine path has been set to " << engineName_ << std::endl;
            return true;
        }
    }
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
    std::cout << "vector buffer size is " << size << std::endl;
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
    if (getFileStatus(engineName_))
    {
        std::cout << "Engine file exists" << std::endl;
    }
    std::ifstream file(engineName_, std::ios::binary | std::ios::ate);
    std::streamsize ssize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(ssize);
    std::cout << "size of binary is " << ssize << std::endl;
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
    const std::vector<cv::Mat>& inputImages, std::vector<std::vector<float>>& featureVectors)
{
    auto inputLength = engine_->getBindingDimensions(0);
    auto outputLength = engine_->getBindingDimensions(1).d[1];
    Dims4 inputDims = {static_cast<int32_t>(inputImages.size()), inputLength.d[1], inputLength.d[2], inputLength.d[3]};
    context_->setBindingDimensions(0, inputDims);
    if (!context_->allInputDimensionsSpecified())
    {
        throw std::runtime_error("All dimensions for inputs are not fully specified");
    }
    auto batchSize = static_cast<int32_t>(inputImages.size());
    // batch size is going to be constant
    inputBuffer_.hostBuffer.resize(inputDims);
    inputBuffer_.deviceBuffer.resize(inputDims);
    Dims2 outputDims{batchSize, outputLength};
    outputBuffer_.hostBuffer.resize(outputDims);
    outputBuffer_.deviceBuffer.resize(outputDims);
    auto* hostBuffer = static_cast<float*>(inputBuffer_.hostBuffer.data());
    for (size_t batch = 0; batch < inputImages.size(); batch++)
    {
        auto image = inputImages[batch];
        image.convertTo(image, CV_32FC3, 1.f / 255.f);
        cv::subtract(image, cv::Scalar(0.5f, 0.5f, 0.5f), image, cv::noArray(), -1);
        cv::divide(image, cv::Scalar(0.5f, 0.5f, 0.5f), image, 1, -1);
        int offset = inputLength.d[1] * inputLength.d[2] * inputLength.d[3] * batch;
        int r = 0, g = 0, b = 0;
        for (int i = 0; i < inputLength.d[1] * inputLength.d[2] * inputLength.d[3]; ++i)
        {
            if (i % 3 == 0)
            {
                hostBuffer[offset + r++] = *(reinterpret_cast<float*>(image.data) + i);
            }
            else if (i % 3 == 1)
            {
                hostBuffer[offset + g++ + inputLength.d[2] * inputLength.d[3]]
                    = *(reinterpret_cast<float*>(image.data) + i);
            }
            else
            {
                hostBuffer[offset + b++ + inputLength.d[2] * inputLength.d[3] * 2]
                    = *(reinterpret_cast<float*>(image.data) + i);
            }
        }
    }
    auto ret = cudaMemcpyAsync(inputBuffer_.deviceBuffer.data(), inputBuffer_.hostBuffer.data(),
        inputBuffer_.hostBuffer.nbBytes(), cudaMemcpyHostToDevice, cudaStream_);
    if (ret != 0)
    {
        return false;
    }
    std::vector<void*> predicitonBindings = {inputBuffer_.deviceBuffer.data(), outputBuffer_.deviceBuffer.data()};
    bool status = context_->enqueueV2(predicitonBindings.data(), cudaStream_, nullptr);
    if (!status)
    {
        return false;
    }
    ret = cudaMemcpyAsync(outputBuffer_.hostBuffer.data(), outputBuffer_.deviceBuffer.data(),
        outputBuffer_.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost, cudaStream_);
    if (ret != 0)
    {
        std::cout << "Unable to copy buffer from GPU back to CPU" << std::endl;
        return false;
    }

    for (int batch = 0; batch < batchSize; ++batch)
    {
        std::vector<float> featureVector;
        featureVector.resize(outputLength);

        memcpy(featureVector.data(),
            reinterpret_cast<const char*>(outputBuffer_.hostBuffer.data()) + batch * outputLength * sizeof(float),
            outputLength * sizeof(float));
        featureVectors.emplace_back(std::move(featureVector));
    }

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

void trt::InferenceEngine::setEngineName(const std::string& engineName)
{

    engineName_ = engineName;
}