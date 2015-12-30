#include "mainwindow.h"
#include "ui_mainwindow.h"

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
}

void MainWindow::setImage() {
  QImage image = camera.getCurrentFrame();
  QImage face = camera.getCurrentFace();
  mainDisplay->setImage(image);
  if (!face.isNull()) {
    faceDisplay->setImage(face);
  }
}

void MainWindow::train() {
  if (trainingTask == nullptr) {
    trainingTask = new TrainingTask();
    connect(trainingTask, SIGNAL(sendMessage(QString)),
            this, SLOT(setLog(QString)));
    connect(trainingTask, SIGNAL(complete(QString)),
            this, SLOT(trainingComplete(QString)));
    trainingTask->start();
  } else {
    setLog("training already started!!");
  }
}

void MainWindow::trainingComplete() {
  if (trainingTask != nullptr) {
    disconnect(trainingTask, SIGNAL(sendMessage(QString)),
            this, SLOT(setLog(QString)));
    disconnect(trainingTask, SIGNAL(complete(QString)),
            this, SLOT(trainingComplete(QString)));
    delete trainingTask;
    trainingTask = nullptr;
    setLog("training complete");
  }
}

void MainWindow::setLog(QString log) {
  ui->logText->append(log);
}
