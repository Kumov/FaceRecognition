#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QPixmap>
#include <QImage>

#include <omp.h>
#include "opencvcamera.h"
#include "imageviewer.h"
#include "classifier.h"

using classifier::FaceClassifier;

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
 private:
  Ui::MainWindow *ui = nullptr;
  OpenCVCamera camera;
  QTimer* timer = nullptr;
  ImageViewer* mainDisplay = nullptr;
  ImageViewer* faceDisplay = nullptr;
  FaceClassifier* classifier = nullptr;
  const int INTERVAL = 33;
};

#endif // MAINWINDOW_H
