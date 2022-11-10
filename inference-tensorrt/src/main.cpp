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
    const std::string InputImage = "/project-workspace/test/U.jpeg";
    auto img = cv::imread(InputImage);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    images.push_back(img);
    images.push_back(img); // batch size is set to 2
    std::vector<std::vector<float>> outputVector;
    auto ret = engine.runInference(images, outputVector);
    std::cout << "size of output vector is " << outputVector[0].size() << std::endl;
    std::cout << "\n" << std::endl;
    for (int i = 0; i < images.size(); i++)
    {
        for (int j = 0; j < outputVector[i].size(); j++)
            std::cout << "\t" << outputVector[i][j];
        std::cout << "next batch values are " << std::endl;
    }
    


    return 0;
}