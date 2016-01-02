#include "trainingtask.h"

TrainingTask::TrainingTask(QString _faceImageDirectory,
                           QString _modelBaseName,
                           QString _modelExtension,
                           double _loadingPercent,
                           FeatureType _featureType) {
  faceImageDirectory = _faceImageDirectory;
  modelBaseName = _modelBaseName;
  modelExtension = _modelExtension;
  loadingPercent = _loadingPercent;
  featureType = _featureType;
}

TrainingTask::~TrainingTask() {
  if (faceClassifier != nullptr) {
    delete faceClassifier;
  }
#ifdef QT_DEBUG
  cout << "release trainer" << endl;
#endif
}

void TrainingTask::run() {
  // create image root directory if not exists
  QDir imageRoot(faceImageDirectory);
  if (!imageRoot.exists()) {
    imageRoot.mkpath(".");
  }

  // get current time stamp
  QDateTime currenTime = QDateTime::currentDateTime();
  currentModelPath = QString(modelBaseName) +
          QString::number(currenTime.toTime_t()) +
          QString(modelExtension);
  sendMessage("current model path: " + currentModelPath);

  sendMessage("loading training data...");
  // prepare loading parameters
  LoadingParams params(faceImageDirectory.toStdString(),
                       loadingPercent,
                       featureType);
  // load the images into matrix
  loadTrainingData(params, trainingData, trainingLabel, names);
  sendMessage("training data loaded");

  // training
  if (faceClassifier == nullptr) {
    sendMessage("creating trainer...");
    faceClassifier = new FaceClassifier(10, 1, 0, 1, 0, 0,
                                        FaceClassifier::C_SVC,
                                        FaceClassifier::RBF,
                                        trainingData,
                                        trainingLabel);
    // connect log message from training task
    connect(faceClassifier, SIGNAL(sendMessage(QString)),
            this, SLOT(captureMessage(QString)));
    sendMessage("training started...");
    faceClassifier->train();
    sendMessage("saving model...");
    faceClassifier->saveModel(currentModelPath.toStdString());
  }

  // prepare the data as QMap<int, QString>
  QMap<int, QString> nameMap;
  for (auto it = names.begin() ; it != names.end() ; it ++) {
    nameMap.insert(it->first, QString(it->second.c_str()));
  }
  complete(currentModelPath, nameMap);
}

void TrainingTask::captureMessage(QString message) {
  sendMessage(message);
}
