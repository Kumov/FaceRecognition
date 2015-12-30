#ifndef TRAININGTASK_H
#define TRAININGTASK_H

#include <QThread>
#include <QString>
#include <QDateTime>
#include <QDir>
#include <QMap>
#include <string>
#include <map>

#include "classifier.h"

using classifier::LoadingParams;
using classifier::loadTrainingData;
using classifier::FaceClassifier;
using cv::Mat;
using std::map;
using std::string;

class TrainingTask : public QThread
{
  Q_OBJECT
public:
  TrainingTask() {}
  virtual ~TrainingTask();
  virtual void run();
signals:
  void sendMessage(QString message);
  void complete(QString modelPath, QMap<int, QString> names);
private:
  QString currentModelPath;
  map<int, string> names;
  FaceClassifier* faceClassifier = nullptr;
  Mat trainingData, trainingLabel;
  const char* FACE_DATA_DIRECTORY = "faces";
  const char* MODEL_BASE_NAME = "facemodel";
  const char* MODEL_EXTENSION = ".xml";
};

#endif // TRAININGTASK_H
