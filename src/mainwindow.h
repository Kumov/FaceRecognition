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
#include <QMessageBox>

#include "opencvcamera.h"
#include "imageviewer.h"
#include "trainingtask.h"
#include "classifier.h"

#define FACE_IMAGE_ROOT_DIR "faces"
#define FACE_MODEL_BASE_NAME "facemodel"
#define FACE_MODEL_EXTENSION ".xml"
#define BACKGROUND_IMAGE_DIR "bg"
#define POSITIVE_DIRECTORY "pos"
#define NEGATIVE_DIRECTORY "neg"
#define IMAGE_OUTPUT_EXTENSION ".jpg"
#define NAME_MAP "names.xml"
#define LIST_NAME "list"
#define ENTRY_NAME "entry"
#define KEY_NAME "key"
#define VALUE_NAME "value"
#define SELECT_TEXT "Select one..."
#define INTERVAL 33
#define PERCENT 0.76

using classifier::FaceClassifier;
using classifier::FeatureType;
using cv::Mat;
using cv::imwrite;

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
  void loadClassifier(QString modelPath);
  void loadNameList();
  void loadNameMap();

 public slots:
  void setImage();
  void train();
  void trainingComplete(QString modelPath,
                        QMap<int, QString> names);
  void takePicture();
  void resume();
  void setLog(QString log);
  void addTrainingData();

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
  const char* BG_IMAGE_DIR = BACKGROUND_IMAGE_DIR;
  const char* POS_DIR = POSITIVE_DIRECTORY;
  const char* NEG_DIR = NEGATIVE_DIRECTORY;
  const char* IMAGE_OUTPUT = IMAGE_OUTPUT_EXTENSION;
  const char* MAPPING_FILE = NAME_MAP;
  const char* LIST = LIST_NAME;
  const char* ENTRY = ENTRY_NAME;
  const char* KEY = KEY_NAME;
  const char* VALUE = VALUE_NAME;
  const char* SELECT = SELECT_TEXT;
  const int CAMEAR_INTERVAL = INTERVAL;
  const double LOADING_PERCENT = PERCENT;
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
