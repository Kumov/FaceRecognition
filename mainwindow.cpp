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
