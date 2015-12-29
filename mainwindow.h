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
  const char* FACE_DATA_DIRECTORY = "faces";
  const int INTERVAL = 33;
};

#endif // MAINWINDOW_H
