#include "MXNetPredictor.h"

#include <iostream>
#include <fstream>
#include <stdexcept>

namespace mx {

std::string readBinaryFile(const std::string &filename) {
    std::ifstream ifile(filename, std::ios::binary | std::ios::in);
    return std::string(std::istreambuf_iterator<char>(ifile),
                       std::istreambuf_iterator<char>());
}

void handleMXNetReturnCode(const int status)
{
    if (status == -1) {
        const char* error = MXGetLastError();
        throw std::runtime_error(error);
    }
}

PredictorHandle loadPredictor(const std::string &symbolPath,
                              const std::string &paramsPath,
                              const mx_uint width,
                              const mx_uint height,
                              const MXNET_DEVICE_TYPE deviceType)
{
    static const int deviceId = 0;
    static const mx_uint numInput = 1;

    static const char* inputKeys[] = {"data"};
    static const mx_uint inputShape[] = {0, 4};
    const mx_uint inputShapeData[] = {numInput, 1, width, height};

    const std::string symbols = readBinaryFile(symbolPath);
    const std::string params = readBinaryFile(paramsPath);

    PredictorHandle handle = nullptr;
    handleMXNetReturnCode(MXPredCreate(
                              symbols.c_str(),
                              params.c_str(),
                              params.size(),
                              deviceType,
                              deviceId,
                              numInput,
                              inputKeys,
                              inputShape,
                              inputShapeData,
                              &handle)
                          );

    return handle;
}

MXNetPredictor::~MXNetPredictor()
{
    MXPredFree(m_predictor);
}

float MXNetPredictor::predict(const cv::Mat2f &input)
{
    handleMXNetReturnCode(MXPredSetInput(
                              m_predictor, "data", reinterpret_cast<float *>(input.data),
                              m_width * m_height * m_channels)
                          );

    handleMXNetReturnCode(MXPredForward(m_predictor));

    static const mx_uint outputIndex = 0;
    static const mx_uint outputBufferSize = 1;

    mx_float outputBuffer[outputBufferSize];

    handleMXNetReturnCode(MXPredGetOutput(
                              m_predictor, outputIndex, outputBuffer, outputBufferSize)
                          );

    return outputBuffer[0];
}

}
