#include "opencvcamera.h"

OpenCVCamera::OpenCVCamera() {
#ifdef QT_DEBUG
  cout << "init VideoCapture" << endl;
#endif

  if (!capture.isOpened()) {
    capture = VideoCapture(0);
  }
}

OpenCVCamera::~OpenCVCamera() {
#ifdef QT_DEBUG
  cout << "close VideoCapture" << endl;
#endif
  capture.release();
}

QImage OpenCVCamera::getCurrentFrame() {
  if (capture.isOpened()) {
    capture.read(frame);
    Mat temp(frame.rows, frame.cols, frame.type());
    cvtColor(frame, temp, CV_BGR2RGB);

    image = QImage((uchar*) temp.data, temp.cols,
                   temp.rows, temp.step, QImage::Format_RGB888);
  } else {
    image = QImage(IMAGE_WIDTH, IMAGE_HEIGHT,
                   QImage::Format_RGB888);
  }
  return image;
}
