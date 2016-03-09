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

using classifier::FaceClassifier;
using classifier::FeatureType;
using cv::Mat;
using cv::Size;

class TrainingTask : public QThread {
  Q_OBJECT
 public:
  TrainingTask(QString _faceImageDirectory,
               QString _modelBaseName,
               QString _modelExtension,
               QString _modelBasePath,
               QString _extraInfoBaseName,
               double lp = (1-classifier::DEFAULT_TEST_PERCENT),
               double s = classifier::DEFAULT_IMAGE_SIZE,
               double ts = classifier::DEFAULT_TRAINING_STEP,
               double g = classifier::DEFAULT_GAMMA,
               FeatureType ft = classifier::LBP);
  virtual ~TrainingTask();
  virtual void run();

 public slots:
  void captureMessage(QString message);

 signals:
  void sendMessage(QString message);
  void complete(QString modelPath,
                QString extraPath,
                QMap<int, QString> names);

 private:
  QString faceImageDirectory, modelBaseName, modelExtension;
  QString modelBasePath, extraInfoBaseName;
  double loadingPercent;
  double trainingStep;
  double defaultGamma;
  FeatureType featureType;
  map<int, string> names;
  FaceClassifier* faceClassifier = nullptr;
  Mat trainingData, trainingLabel;
  Size trainingSize;
};

#endif // TRAININGTASK_H
