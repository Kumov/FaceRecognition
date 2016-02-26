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
  const unsigned int LBP_FEATURE_LENGTH = 256;
  const unsigned int LTP_FEATURE_LENGTH = 9841;
  const unsigned int CSLTP_FEATURE_LENGTH = 121;
  void changeBrightness(Mat& image, double alpha);
  void changeBrightness(Mat& image, double alpha, double beta);
  void rotateImage(Mat& image, const double deg);
  void computeLBP(Mat& image, Mat& lbp);
  void computeLTP(Mat& image, Mat& ltp, int threshold);
  void computeCSLTP(Mat& image, Mat& csltp, int threshold);
  void computeHaar(Mat& image, Mat& haar,
                   unsigned int& featureLength);
}

#endif /* end of include guard: PROCESS_H */
