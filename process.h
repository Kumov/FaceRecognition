#ifndef PROCESS_H
#define PROCESS_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace process {
  void changeBrightness(cv::Mat& image, double alpha);
  void changeBrightness(cv::Mat& image, double alpha, double beta);
  void rotateImage(cv::Mat& image, const double deg);
  void computeLBP(cv::Mat& image, cv::Mat& lbp);
}

#endif /* end of include guard: PROCESS_H */
