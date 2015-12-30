#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QPixmap>
#include <QImage>
#include <QDir>
#include <QString>
#include <QDateTime>

#include <omp.h>
#include "opencvcamera.h"
#include "imageviewer.h"
#include "classifier.h"

using classifier::FaceClassifier;
using classifier::loadTrainingData;
using classifier::LoadingParams;
using std::map;
using std::string;
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
 private:
  void setupTraining();
  Ui::MainWindow *ui = nullptr;
  OpenCVCamera camera;
  QTimer* timer = nullptr;
  ImageViewer* mainDisplay = nullptr;
  ImageViewer* faceDisplay = nullptr;
  FaceClassifier* classifier = nullptr;
  QString currentModelPath;
  Mat trainingData, trainingLabel;
  map<int, string> names;
  const char* FACE_DATA_DIRECTORY = "faces";
  const int INTERVAL = 33;
};

#endif // MAINWINDOW_H
