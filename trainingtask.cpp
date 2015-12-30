#include "trainingtask.h"

TrainingTask::~TrainingTask() {
  if (faceClassifier != nullptr) {
    delete faceClassifier;
  }
#ifdef QT_DEBUG
  cout << "release trainer" << endl;
#endif
}

void TrainingTask::run() {
  QDir imageRoot(FACE_DATA_DIRECTORY);
  if (!imageRoot.exists()) {
    imageRoot.mkpath(".");
  }
  QDateTime currenTime = QDateTime::currentDateTime();
  currentModelPath = QString(MODEL_BASE_NAME) +
          QString::number(currenTime.toTime_t()) +
          QString(MODEL_EXTENSION);
  sendMessage("current model path: " + currentModelPath);

  sendMessage("loading training data...");
  LoadingParams params(FACE_DATA_DIRECTORY, 0.9, classifier::LBP);
  loadTrainingData(params, trainingData, trainingLabel, names);
  sendMessage("training data loaded");
  cout << trainingData;

  if (faceClassifier == nullptr) {
    sendMessage("creating trainer...");
    faceClassifier = new FaceClassifier(10, 1, 0, 1, 0, 0,
                                        classifier::C_SVC,
                                        classifier::RBF,
                                        trainingData,
                                        trainingLabel);
    sendMessage("training started...");
    faceClassifier->train();
    sendMessage("saving model...");
    faceClassifier->saveModel(currentModelPath.toStdString());
  }

  QMap<int, QString> nameMap;
  for (auto it = names.begin() ; it != names.end() ; it ++) {
    nameMap.insert(it->first, QString(it->second.c_str()));
  }
  complete(currentModelPath, nameMap);
}
