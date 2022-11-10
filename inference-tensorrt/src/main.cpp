#include "inference_helper.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "opencv4/opencv2/opencv.hpp"

static void helper(std::string name)
{
    std::cerr << "Usage :  " << name << "\t"
              << "--enginePath "
              << " "
              << "/home/asif/someImage" << std::endl;
}

/*TODO:
- parse values as argument to main function
- Parse and use it
- Move the checks into functions
*/
int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Arguments cannot be less than two" << std::endl;
        helper(argv[0]);
    }
    std::vector<std::string> allArgs(argv, argv + argc);
    assert(allArgs[1] == "--enginePath");
    trt::Settings settings;
    settings.optBatchSize = {2};
    trt::InferenceEngine engine(settings);
    engine.setEngineName(allArgs[2]);
    const std::string OnnxModelPath = "onnx/dynamic_restnet50-tuned.onnx";
    engine.buildNetwork(OnnxModelPath, false);
    engine.loadNetwork();
    std::vector<cv::Mat> images;
    const std::string InputImage = "/test/U.jpeg";
    auto img = cv::imread(InputImage);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    return 0;
}