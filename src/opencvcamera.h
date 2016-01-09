#ifndef OPENCVCAMERA_H
#define OPENCVCAMERA_H

#include <QImage>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#define DEFAULT_WIDTH 640
#define DEFAULT_HEIGHT 480

#if defined(__unix__)
#define OBJECT_DETECT_MODEL "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml"
#elif defined(__WIN32)
#define OBJECT_DETECT_MODEL "./haarcascade_frontalface_alt2.xml"
#endif

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
  void getCurrentFaceMat(Mat& face);

 private:
  VideoCapture capture;
  CascadeClassifier faceFinder;
  Mat frame, main, face;
  QImage mainImage, faceImage;
  const int IMAGE_WIDTH = DEFAULT_WIDTH;
  const int IMAGE_HEIGHT = DEFAULT_HEIGHT;
  const char* FACE_FINDER_MODEL = OBJECT_DETECT_MODEL;
};

#undef DEFAULT_WIDTH
#undef DEFAULT_HEIGHT
#undef OBJECT_DETECT_MODEL

#endif // OPENCVCAMERA_H
