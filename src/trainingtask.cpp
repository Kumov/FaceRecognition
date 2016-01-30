#include "trainingtask.h"

#ifdef QT_DEBUG
using std::cout;
using std::endl;
#endif

using classifier::LoadingParams;
using classifier::FaceClassifierParams;
using classifier::TrainingDataLoader;
using std::map;
using std::string;

TrainingTask::TrainingTask(QString _faceImageDirectory,
                           QString _modelBaseName,
                           QString _modelExtension,
                           QString _modelBasePath,
                           double _loadingPercent,
                           double _size,
                           double _trainingStep,
                           double _gamma,
                           FeatureType _featureType) {
  faceImageDirectory = _faceImageDirectory;
  modelBaseName = _modelBaseName;
  modelExtension = _modelExtension;
  modelBasePath = _modelBasePath;
  loadingPercent = _loadingPercent;
  trainingStep = _trainingStep;
  trainingSize = Size(_size, _size);
  defaultGamma = _gamma;
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

  // get current time stamp and compute current model path
  QDateTime currenTime = QDateTime::currentDateTime();

  // create the base directory if not exist
  QDir modelBaseDirectory(modelBasePath);
  if (!modelBaseDirectory.exists()) {
    modelBaseDirectory.mkpath(".");
  }
  currentModelPath = modelBasePath +
      QDir::separator() +
      QString(modelBaseName) +
      QString::number(currenTime.toTime_t()) +
      QString(modelExtension);
  sendMessage("current model path: " + currentModelPath);

  sendMessage("loading training data...");
  // prepare loading parameters
  LoadingParams params(faceImageDirectory.toStdString(),
                       loadingPercent,
                       featureType, trainingSize);
  // load the images into matrix
  TrainingDataLoader loader(params);
  connect(&loader, SIGNAL(sendMessage(QString)), this,
          SLOT(captureMessage(QString)));
  loader.load(trainingData, trainingLabel, names);
  // old way to load data
  // loadTrainingData(params, trainingData, trainingLabel, names);
  sendMessage("training data loaded");

  // training
  if (faceClassifier == nullptr) {
    sendMessage("creating trainer...");
    FaceClassifierParams classifierParam(trainingSize,
                                         defaultGamma, trainingStep,
                                         1.0 - loadingPercent);
    faceClassifier = new FaceClassifier(classifierParam,
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
