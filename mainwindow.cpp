#include "mainwindow.h"
#include "ui_mainwindow.h"

#define MODEL_BASE_NAME "facemodel"
#define MODEL_EXTENSION ".xml"

MainWindow::MainWindow(QWidget *parent) :
            QMainWindow(parent),
            ui(new Ui::MainWindow),
            timer(new QTimer()),
            mainDisplay(new ImageViewer()),
            faceDisplay(new ImageViewer()) {
  ui->setupUi(this);

  ui->mainWidget->layout()->addWidget(mainDisplay);
  ui->faceLayout->addWidget(faceDisplay);

  timer->start(INTERVAL);
  connect(timer, SIGNAL(timeout()), this, SLOT(setImage()));
  connect(ui->trainButton, SIGNAL(pressed()), SLOT(train()));
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
  if (classifier != nullptr)
    delete classifier;
}

void MainWindow::setImage() {
  ui->logText->append("Capturing Image from Camera");

  QImage image = camera.getCurrentFrame();
  QImage face = camera.getCurrentFace();
  mainDisplay->setImage(image);
  if (!face.isNull()) {
    faceDisplay->setImage(face);
  }
}

void MainWindow::train() {
  QDir imageRoot(FACE_DATA_DIRECTORY);
  if (!imageRoot.exists()) {
    imageRoot.mkpath(".");
  }
  setupTraining();
}

void MainWindow::setupTraining() {
  QDateTime currenTime = QDateTime::currentDateTime();
  currentModelPath = QString(MODEL_BASE_NAME) +
          QString::number(currenTime.toTime_t()) +
          QString(MODEL_EXTENSION);
  cout << currentModelPath.toStdString() << endl;
}

#undef MODEL_BASE_NAME
#undef MODEL_EXTENSION
