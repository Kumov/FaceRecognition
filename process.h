#ifndef PROCESS_H
#define PROCESS_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#ifdef DEBUG
#include <iostream>
using std::cout;
using std::endl;
#endif

using cv::Mat;

namespace process {
  void changeBrightness(Mat& image, double alpha);
  void changeBrightness(Mat& image, double alpha, double beta);
  void rotateImage(Mat& image, const double deg);
  void computeLBP(Mat& image, Mat& lbp);
  void computeLTP(Mat& image, Mat& ltp, int threshold);
  void computeCSLTP(Mat& image, Mat& csltp,
                    int threshold, uint32_t level);
}

#endif /* end of include guard: PROCESS_H */
