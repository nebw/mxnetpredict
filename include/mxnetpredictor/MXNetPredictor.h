#pragma once

#include <string>

#include <opencv2/core/core.hpp>

namespace mx {
typedef unsigned int mx_uint;
typedef void *PredictorHandle;

enum MXNET_DEVICE_TYPE {
    CPU = 1,
    GPU = 2
};

std::string readBinaryFile(const std::string& filename);

void handleMXNetReturnCode(const int status);

PredictorHandle loadPredictor(const std::string& symbolPath,
                              const std::string& paramsPath,
                              const mx_uint width,
                              const mx_uint height,
                              const MXNET_DEVICE_TYPE deviceType);

class MXNetPredictor {
public:
    MXNetPredictor(const std::string& symbol,
              const std::string& params,
              const size_t width,
              const size_t height,
              const MXNET_DEVICE_TYPE deviceType = MXNET_DEVICE_TYPE::CPU)
        : m_width(width)
        , m_height(height)
        , m_predictor(loadPredictor(symbol, params, width, height, deviceType))
    {}

    virtual ~MXNetPredictor();

    float predict(const cv::Mat &input);

private:
    static const size_t m_channels = 1;
    const size_t m_width;
    const size_t m_height;
    PredictorHandle m_predictor;
};

}
