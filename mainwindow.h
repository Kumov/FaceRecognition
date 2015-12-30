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
  void trainingComplete(QString modelPath, QMap<int, QString> names);
  void setLog(QString log);
 private:
  Ui::MainWindow *ui = nullptr;
  OpenCVCamera camera;
  QTimer* timer = nullptr;
  ImageViewer* mainDisplay = nullptr;
  ImageViewer* faceDisplay = nullptr;
  TrainingTask* trainingTask = nullptr;
  const int INTERVAL = 33;
};

#endif // MAINWINDOW_H
