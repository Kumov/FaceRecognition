#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QPixmap>
#include <QImage>
#include <QString>
#include <QMap>

#include "opencvcamera.h"
#include "imageviewer.h"
#include "trainingtask.h"
#include "classifier.h"

#define FACE_IMAGE_ROOT_DIR "faces"
#define FACE_MODEL_BASE_NAME "facemodel"
#define FACE_MODEL_EXTENSION ".xml"
#define INTERVAL 33
#define PERCENT 0.9

using classifier::FaceClassifier;
using classifier::FeatureType;
using cv::Mat;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow {
  Q_OBJECT
 public:
  explicit MainWindow(QWidget *parent = 0);
  ~MainWindow();
 public slots:
  void setImage();
  void train();
  void trainingComplete(QString modelPath,
                        QMap<int, QString> names);
  void takePicture();
  void resume();
  void setLog(QString log);
 private:
  Ui::MainWindow *ui = nullptr;
  OpenCVCamera camera;
  QTimer* timer = nullptr;
  ImageViewer* mainDisplay = nullptr;
  ImageViewer* faceDisplay = nullptr;
  TrainingTask* trainingTask = nullptr;
  FaceClassifier* faceClassifier = nullptr;
  QMap<int, QString> names;
  Mat face;
  bool pictureTaken = false;
  const char* FACE_IMAGE_DIR = FACE_IMAGE_ROOT_DIR;
  const char* MODEL_BASE_NAME = FACE_MODEL_BASE_NAME;
  const char* MODEL_EXTENSION = FACE_MODEL_EXTENSION;
  const int CAMEAR_INTERVAL = INTERVAL;
  const double LOADING_PERCENT = PERCENT;
  const FeatureType FEATURE_TYPE = classifier::LBP;
};

#undef FACE_IMAGE_ROOT_DIR
#undef INTERVAL

#endif // MAINWINDOW_H
