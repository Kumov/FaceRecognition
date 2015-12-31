#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QPixmap>
#include <QImage>
#include <QString>
#include <QMap>
#include <QFile>
#include <QXmlStreamWriter>
#include <QXmlStreamReader>

#include "opencvcamera.h"
#include "imageviewer.h"
#include "trainingtask.h"
#include "classifier.h"

#define FACE_IMAGE_ROOT_DIR "faces"
#define FACE_MODEL_BASE_NAME "facemodel"
#define FACE_MODEL_EXTENSION ".xml"
#define NAME_MAP "names.xml"
#define LIST_NAME "list"
#define ENTRY_NAME "entry"
#define KEY_NAME "key"
#define VALUE_NAME "value"
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
  void writeMap();
  void readMap();
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
  const char* MAPPING_FILE = NAME_MAP;
  const char* LIST = LIST_NAME;
  const char* ENTRY = ENTRY_NAME;
  const char* KEY = KEY_NAME;
  const char* VALUE = VALUE_NAME;
  const int CAMEAR_INTERVAL = INTERVAL;
  const double LOADING_PERCENT = PERCENT;
  const FeatureType FEATURE_TYPE = classifier::LBP;
};

#undef FACE_IMAGE_ROOT_DIR
#undef FACE_MODEL_BASE_NAME
#undef FACE_MODEL_EXTENSION
#undef NAME_MAP
#undef LIST_NAME
#undef ENTRY_NAME
#undef KEY_NAME
#undef VALUE_NAME
#undef INTERVAL
#undef PERCENT

#endif // MAINWINDOW_H
