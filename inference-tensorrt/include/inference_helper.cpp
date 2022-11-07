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

