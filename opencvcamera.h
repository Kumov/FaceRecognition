#ifndef OPENCVCAMERA_H
#define OPENCVCAMERA_H

#include <QImage>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#ifdef QT_DEBUG
#include <iostream>
using std::cout;
using std::endl;
#endif

using std::vector;
using cv::VideoCapture;
using cv::Mat;
using cv::Rect;
using cv::cvtColor;
using cv::CascadeClassifier;

class OpenCVCamera {
 public:
  OpenCVCamera();
  ~OpenCVCamera();
  QImage getCurrentFrame();
  QImage getCurrentFace();
 private:
  VideoCapture capture;
  CascadeClassifier faceFinder;
  Mat frame, main, face;
  QImage mainImage, faceImage;
  const int IMAGE_WIDTH = 640;
  const int IMAGE_HEIGHT = 480;
  const char* FACE_FINDER_MODEL = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml";
};

#endif // OPENCVCAMERA_H
