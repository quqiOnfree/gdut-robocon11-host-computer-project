#pragma once

#include "geometry_utils.hpp"
#include "yolo_model_classifier.hpp"
#include "yolo_model_detector.hpp"
#include "yolo_model_segmentor.hpp"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class BlockFacePipeline {
public:
  // 分类的结果：这张面属于哪个类别
  struct FaceResult {
    cv::Mat warped_face; // 透视矫正后的正方形面图像
    int class_id = -1;
    std::string class_name;
    float confidence = 0.0f;
    cv::Rect source_bbox; // 在原图中检测框的位置（用于索引）
    int face_index = 0;   // 该检测框内的第几个面
  };

  // 构造函数：传入三个模型的路径和对应的类别名
  BlockFacePipeline(const std::string &detect_model_path,
                    const std::vector<std::string> &detect_classes,
                    const std::string &segment_model_path,
                    const std::string &classify_model_path,
                    const std::vector<std::string> &face_classes,
                    int warp_size = 512, float detect_conf = 0.5f,
                    float min_face_area = 500)
      : detector_(detect_model_path, detect_classes),
        segmentor_(segment_model_path),
        classifier_(classify_model_path, face_classes), warp_size_(warp_size),
        detect_conf_(detect_conf), min_face_area_(min_face_area) {}

  // 处理一张图像，返回所有检测到的面的分类结果
  std::vector<FaceResult> process(const cv::Mat &image) {
    std::vector<FaceResult> results;

    // 1. 检测方块区域
    auto detections = detector_.process(image, detect_conf_);
    for (size_t i = 0; i < detections.size(); ++i) {
      const auto &det = detections[i];
      // 可以稍微扩大裁剪区域，避免边缘切掉
      cv::Rect roi = det.box;
      int pad = 5;
      roi.x = std::max(0, roi.x - pad);
      roi.y = std::max(0, roi.y - pad);
      roi.width = std::min(image.cols - roi.x, roi.width + 2 * pad);
      roi.height = std::min(image.rows - roi.y, roi.height + 2 * pad);
      if (roi.width <= 0 || roi.height <= 0)
        continue;

      cv::Mat cropped = image(roi).clone();

      // 2. 对裁剪区域做分割，得到 mask（尺寸与 cropped 相同）
      cv::Mat mask =
          segmentor_.process(cropped); // 返回 CV_8UC1，每个像素为类别 id
      if (mask.empty())
        continue;

      // 3. 从 mask 中提取每一个面的轮廓（假设每个面是不同区域，且类别 id
      // 不同或区域隔开） 这里简单把 mask
      // 转为二值图，然后找所有轮廓，每个轮廓对应一个面。
      cv::Mat binary;
      cv::threshold(mask, binary, 1, 255, cv::THRESH_BINARY); // 所有非背景
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(binary, contours, cv::RETR_EXTERNAL,
                       cv::CHAIN_APPROX_SIMPLE);

      for (size_t j = 0; j < contours.size(); ++j) {
        double area = cv::contourArea(contours[j]);
        if (area < min_face_area_)
          continue;

        // 近似为四边形
        auto quad = approx_to_quad(contours[j]);
        if (quad.size() != 4)
          continue;

        // 4. 透视矫正为正正方形
        cv::Mat warped = warp_polygon_to_square(cropped, quad, warp_size_);

        // 5. 分类
        auto cls_res = classifier_.process(warped);
        if (cls_res.classId < 0)
          continue; // 分类失败

        FaceResult fr;
        fr.warped_face = warped;
        fr.class_id = cls_res.classId;
        fr.class_name = cls_res.className;
        fr.confidence = cls_res.confidence;
        fr.source_bbox = det.box;
        fr.face_index = static_cast<int>(j);
        results.push_back(fr);
      }
    }
    return results;
  }

private:
  YoloOnnxDetector detector_;
  YoloOnnxSegmentor segmentor_;
  YoloOnnxClassifier classifier_;
  int warp_size_;
  float detect_conf_;
  float min_face_area_;
};
