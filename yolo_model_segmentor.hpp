#pragma once

#include "yolo_model_interface.hpp"

class YoloOnnxSegmentor : public YoloOnnxProcessor<YoloOnnxSegmentor> {
  friend class YoloOnnxProcessor<YoloOnnxSegmentor>;

public:
  // 返回的 mask 尺寸与原图相同，值为类别 id
  YoloOnnxSegmentor(const std::string &modelPath)
      : YoloOnnxProcessor<YoloOnnxSegmentor>(modelPath) {}

  static cv::Mat drawMask(const cv::Mat &image, const cv::Mat &mask,
                          const std::vector<cv::Vec3b> &palette) {
    cv::Mat colorMask(image.size(), CV_8UC3);
    for (int r = 0; r < mask.rows; ++r) {
      for (int c = 0; c < mask.cols; ++c) {
        int id = mask.at<uchar>(r, c);
        if (id < palette.size())
          colorMask.at<cv::Vec3b>(r, c) = palette[id];
        else
          colorMask.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
      }
    }
    cv::Mat blended;
    cv::addWeighted(image, 0.6, colorMask, 0.4, 0, blended);
    return blended;
  }

private:
  cv::Mat processImpl(const cv::Mat &frame, float /*unused*/) {
    float ratio, dw, dh;
    auto inputTensor = preprocess(frame, ratio, dw, dh);
    auto outputs = runInference(inputTensor);
    if (outputs.empty() || !outputs[0].IsTensor())
      return {};

    auto outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    // 假设输出 [1, C, H, W]
    if (outShape.size() != 4 || outShape[0] != 1)
      return {};

    int C = static_cast<int>(outShape[1]);
    int outH = static_cast<int>(outShape[2]);
    int outW = static_cast<int>(outShape[3]);
    const float *outData = outputs[0].GetTensorData<float>();

    // 每个像素取 argmax
    cv::Mat labelMask(outH, outW, CV_8UC1);
    for (int y = 0; y < outH; ++y) {
      for (int x = 0; x < outW; ++x) {
        const float *pixel = outData + (y * outW + x) * C;
        int maxId = std::max_element(pixel, pixel + C) - pixel;
        labelMask.at<uchar>(y, x) = static_cast<uchar>(maxId);
      }
    }

    // 计算 letterbox 中的有效区域（去掉灰边）
    int top = static_cast<int>(std::round(dh - 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int cropH = static_cast<int>(std::round(frame.rows * ratio));
    int cropW = static_cast<int>(std::round(frame.cols * ratio));
    cv::Rect roi(left, top, cropW, cropH);
    cv::Mat cropped = labelMask(roi);

    cv::Mat mask;
    cv::resize(cropped, mask, cv::Size(frame.cols, frame.rows), 0, 0,
               cv::INTER_NEAREST);
    return mask;
  }
};
