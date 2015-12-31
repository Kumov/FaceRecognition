#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
            QMainWindow(parent),
            ui(new Ui::MainWindow),
            timer(new QTimer()),
            mainDisplay(new ImageViewer()),
            faceDisplay(new ImageViewer()),
            faceClassifier(new FaceClassifier()) {
  ui->setupUi(this);

  ui->mainWidget->layout()->addWidget(mainDisplay);
  ui->faceLayout->addWidget(faceDisplay);

  timer->start(CAMEAR_INTERVAL);
  connect(timer, SIGNAL(timeout()), this, SLOT(setImage()));
  connect(ui->trainButton, SIGNAL(pressed()), SLOT(train()));
  connect(ui->selectButton, SIGNAL(pressed()), this, SLOT(takePicture()));
  connect(ui->resumeButton, SIGNAL(pressed()), this, SLOT(resume()));

  // scan image directory for people list
  QDir imageRoot(FACE_IMAGE_DIR);
  imageRoot.setFilter(QDir::Dirs);
  QStringList dirList = imageRoot.entryList();
  for (int i = 0 ; i < dirList.count() ; i ++) {
    if (dirList[i] != "." && dirList[i] != "..")
      ui->selectComboBox->addItem(dirList[i]);
  }

  // register type for signal and slot
  qRegisterMetaType<QMap<int, QString> >();
}

MainWindow::~MainWindow() {
  if (timer != nullptr)
    delete timer;
  if (ui != nullptr)
    delete ui;
  if (mainDisplay != nullptr)
    delete mainDisplay;
  if (faceDisplay != nullptr)
    delete faceDisplay;
  if (trainingTask)
    delete trainingTask;
  if (faceClassifier != nullptr)
    delete faceClassifier;
}

void MainWindow::setImage() {
  // display main image
  QImage image = camera.getCurrentFrame();
  mainDisplay->setImage(image);

  // if the picture is not taken constantly shot the face
  if (!pictureTaken) {
    QImage face = camera.getCurrentFace();
    if (!face.isNull()) {
      faceDisplay->setImage(face);
      camera.getCurrentFaceMat(this->face);
    }
  }
}

void MainWindow::train() {
  // start training task asynchroniously
  if (trainingTask == nullptr) {
    trainingTask = new TrainingTask(FACE_IMAGE_DIR,
                                    MODEL_BASE_NAME,
                                    MODEL_EXTENSION,
                                    LOADING_PERCENT,
                                    FEATURE_TYPE);
    connect(trainingTask, SIGNAL(sendMessage(QString)),
            this, SLOT(setLog(QString)));
    connect(trainingTask,
            SIGNAL(complete(QString, QMap<int, QString>)),
            this,
            SLOT(trainingComplete(QString, QMap<int, QString>)));
    trainingTask->start();
  } else {
    setLog("training already started!!");
  }
}

void MainWindow::trainingComplete(QString modelPath,
                                  QMap<int, QString> names) {
  // clean up after training task complete
  // and retrieve name map and model path
  if (trainingTask != nullptr) {
    disconnect(trainingTask, SIGNAL(sendMessage(QString)),
            this, SLOT(setLog(QString)));
    disconnect(trainingTask,
            SIGNAL(complete(QString, QMap<int, QString>)),
            this,
            SLOT(trainingComplete(QString, QMap<int, QString>)));
    delete trainingTask;
    trainingTask = nullptr;
    setLog("training complete");
    setLog("new model written: " + modelPath);
    // copy to target
    QString target = QString(MODEL_BASE_NAME) + QString(MODEL_EXTENSION);
    if (QFile::exists(target)) {
      QFile::remove(target);
    }
    QFile::copy(modelPath, target);
    setLog("new model copied");

    // load the new model
    faceClassifier->load(target);
    setLog("new model loaded");

    this->names = names;
    QMapIterator<int, QString> it(names);
    while (it.hasNext()) {
      it.next();
      QString index = QString::number(it.key());
      QString name = it.value();
      setLog(name + ": " + index);
    }
  }
}

void MainWindow::setLog(QString log) {
  ui->logText->append(log);
}

void MainWindow::takePicture() {
  pictureTaken = true;
  if (faceClassifier != nullptr) {
    if (!faceClassifier->isLoaded()) {
      QString modelPath = QString(MODEL_BASE_NAME) +
          QString(MODEL_EXTENSION);
      faceClassifier->load(modelPath.toStdString());
      setLog("loading face classifier " + modelPath + "...");
    }

    int result = faceClassifier->predictImageSample(this->face);

    setLog("result: " + QString::number(result));

    QMapIterator<int, QString> it(names);
    while (it.hasNext()) {
      it.next();
      if (it.key() == result) {
        ui->whoLabel->setText("Are you " + it.value());
        break;
      }
    }
  }

}

void MainWindow::resume() {
  pictureTaken = false;
}
