#include "opencvcamera.h"

OpenCVCamera::OpenCVCamera() {
#ifdef QT_DEBUG
  cout << "starting VideoCapture" << endl;
#endif
  if (!capture.isOpened()) {
    capture = VideoCapture(0);
  }
#ifdef QT_DEBUG
  cout << "init VideoCapture" << endl;
#endif
  faceFinder = CascadeClassifier(FACE_FINDER_MODEL);
#ifdef QT_DEBUG
  cout << "init Classifier" << endl;
#endif
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
    frame.copyTo(main);
    cvtColor(main, main, CV_BGR2RGB);

    mainImage = QImage((uchar*) main.data,
                   main.cols, main.rows,
                   main.step, QImage::Format_RGB888);
  } else {
    mainImage = QImage(IMAGE_WIDTH, IMAGE_HEIGHT,
                   QImage::Format_RGB888);
  }
  return mainImage;
}

QImage OpenCVCamera::getCurrentFace() {
  if (frame.data) {
    vector<Rect> faces;
    faceFinder.detectMultiScale(frame, faces);
    if (faces.size() > 0) {
      frame(faces[0]).copyTo(face);
      cvtColor(face, face, CV_BGR2RGB);

      faceImage = QImage(face.data,
                         face.cols, face.rows,
                         face.step, QImage::Format_RGB888);
    } else {
      faceImage = QImage();
    }
  } else {
    faceImage = QImage();
  }
  return faceImage;
}
