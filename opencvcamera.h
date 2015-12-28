#ifndef OPENCVCAMERA_H
#define OPENCVCAMERA_H

#include <QImage>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#ifdef QT_DEBUG
#include <iostream>
using std::cout;
using std::endl;
#endif

using cv::VideoCapture;
using cv::Mat;
using cv::cvtColor;

class OpenCVCamera {
 public:
  OpenCVCamera();
  ~OpenCVCamera();
  QImage getCurrentFrame();
 private:
  VideoCapture capture;
  Mat frame;
  QImage image;
  const int IMAGE_WIDTH = 640;
  const int IMAGE_HEIGHT = 480;
};

#endif // OPENCVCAMERA_H
