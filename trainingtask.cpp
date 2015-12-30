#include "trainingtask.h"

TrainingTask::TrainingTask()
{
  faceClassifier = new FaceClassifier(1, 1, 1, 3, 1, 1,
                                      classifier::C_SVC,
                                      classifier::LINEAR);
}

TrainingTask::~TrainingTask() {
  if (faceClassifier != nullptr) {
    delete faceClassifier;
  }
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

  LoadingParams params(FACE_DATA_DIRECTORY, 0.9, classifier::LBP);
  loadTrainingData(params, trainingData, trainingLabel, names);
  sendMessage("training data loaded");

  complete();
}
