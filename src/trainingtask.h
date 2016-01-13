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
using classifier::FaceClassifier;
using classifier::FeatureType;
using classifier::TrainingDataLoader;
using cv::Mat;
using cv::Size;
using std::map;
using std::string;

class TrainingTask : public QThread {
  Q_OBJECT
 public:
  TrainingTask(QString _faceImageDirectory,
               QString _modelBaseName,
               QString _modelExtension,
               QString _modelBasePath,
               double _loadingPercent,
               FeatureType _featureType = classifier::LBP);
  virtual ~TrainingTask();
  virtual void run();

 public slots:
  void captureMessage(QString message);

 signals:
  void sendMessage(QString message);
  void complete(QString modelPath, QMap<int, QString> names);

 private:
  QString faceImageDirectory, modelBaseName, modelExtension;
  QString currentModelPath, modelBasePath;
  double loadingPercent;
  FeatureType featureType;
  map<int, string> names;
  FaceClassifier* faceClassifier = nullptr;
  Mat trainingData, trainingLabel;
};

#endif // TRAININGTASK_H
