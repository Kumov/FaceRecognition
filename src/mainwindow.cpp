#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
            QMainWindow(parent),
            ui(new Ui::MainWindow),
            timer(new QTimer()),
            mainDisplay(new ImageViewer()),
            faceDisplay(new ImageViewer()),
            faceClassifier(new FaceClassifier()) {
  // main ui setup
  ui->setupUi(this);

  // add display widget
  ui->mainWidget->layout()->addWidget(mainDisplay);
  ui->faceLayout->addWidget(faceDisplay);

  // init
  ui->rbLBP->setChecked(true);

  // start timer to update frame
  timer->start(CAMEAR_INTERVAL);
  connect(timer, SIGNAL(timeout()), this, SLOT(setImage()));
  // button event
  connect(ui->trainButton, SIGNAL(pressed()),
          this, SLOT(train()));
  connect(ui->selectButton, SIGNAL(pressed()),
          this, SLOT(takePicture()));
  connect(ui->resumeButton, SIGNAL(pressed()),
          this, SLOT(resume()));
  connect(ui->yesButton, SIGNAL(pressed()),
          this, SLOT(resume()));
  connect(ui->noButton, SIGNAL(pressed()),
          this, SLOT(addTrainingData()));
  // face classifier message capture
  connect(faceClassifier, SIGNAL(sendMessage(QString)),
          this, SLOT(setLog(QString)));

  // load peoples name
  loadNameList();
  // load the mapping of peoples name
  loadNameMap();

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
    FeatureType featureType = classifier::LBP;
    if (ui->rbLBP->isChecked()) {
      featureType = classifier::LBP;
    } else if (ui->rbLTP->isChecked()) {
      featureType = classifier::LTP;
    } else if (ui->rbCSLTP->isChecked()) {
      featureType = classifier::CSLTP;
    }
    trainingTask = new TrainingTask(FACE_IMAGE_DIR,
                                    MODEL_BASE_NAME,
                                    MODEL_EXTENSION,
                                    LOADING_PERCENT,
                                    featureType);
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
    loadClassifier(target);

    // output new name map
    this->names = names;
    QMapIterator<int, QString> it(names);
    while (it.hasNext()) {
      it.next();
      QString index = QString::number(it.key());
      QString name = it.value();
      setLog(name + ": " + index);
    }
    // write to file
    this->writeMap();
    setLog("new name mapping written");
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
      this->loadClassifier(modelPath);
      setLog("loading face classifier " + modelPath + "...");
    }

    int result = faceClassifier->predictImageSample(this->face);

    setLog("result: " + QString::number(result));

    QMapIterator<int, QString> it(names);
    while (it.hasNext()) {
      it.next();
      if (it.key() == result) {
        if (it.value() == QString(BG_IMAGE_DIR)) {
          ui->whoLabel->setText("We don't recognize you!!!");
        } else {
          ui->whoLabel->setText("Are you " + it.value());
        }
        break;
      }
    }
  }
}

void MainWindow::resume() {
  pictureTaken = false;
  setLog("continue to capture camera image");
}

void MainWindow::writeMap() {
  // open output file
  QFile nameMapFile(MAPPING_FILE);

  if (nameMapFile.open(QIODevice::WriteOnly)) {
    QXmlStreamWriter out(&nameMapFile);
    out.setAutoFormatting(true);
    out.setAutoFormattingIndent(2);
    out.writeStartDocument();
    out.writeStartElement(LIST);

    QMapIterator<int, QString> it(names);
    while (it.hasNext()) {
      it.next();
      out.writeStartElement(ENTRY);
      out.writeTextElement(QString(KEY), QString::number(it.key()));
      out.writeTextElement(QString(VALUE), it.value());
      out.writeEndElement();
    }
    out.writeEndElement();
    out.writeEndDocument();
  }
}

void MainWindow::readMap() {
  // clear the old map
  names = QMap<int, QString>();

  QFile nameMapFile(MAPPING_FILE);
  if (nameMapFile.open(QIODevice::ReadOnly)) {
    QXmlStreamReader in;
    in.setDevice(&nameMapFile);
    in.readNext();

    while (!in.atEnd()) {
      if (in.isStartElement() && in.name() == QString(LIST)) {
        in.readNext();
        while (!in.atEnd()) {
          in.readNext();
          if (in.isStartElement() && in.name() == QString(ENTRY)) {
            in.readNext();
            QString key, value;
            while (!in.atEnd()) {
              in.readNext();
              if (in.isStartElement() && in.name() == QString(KEY)) {
                key = in.readElementText();
                in.readNext();
              } else if (in.isStartElement() && in.name() == QString(VALUE)) {
                value = in.readElementText();
                in.readNext();
              } else if (in.isEndElement()) {
                in.readNext();
                break;
              } else {
                in.readNext();
              }
            }
            names.insert(key.toInt(), value);
          } else if (in.isEndElement()) {
            in.readNext();
            break;
          }
        }
      }
      in.readNext();
    }
  } else {
    setLog("no name mapping xml file found");
  }
}

void MainWindow::addTrainingData() {
  // check if user already took the picture where the capturing stops
  if (pictureTaken) {
    // check if user select the target user to add sample
    QString user = ui->selectComboBox->currentText();
    if (user != QString(SELECT)) {
      size_t index = 0;
      if (user != QString(BG_IMAGE_DIR)) {
        // if target is a user image
        // calculate the path
        QString filePath = QString(FACE_IMAGE_DIR) + QDir::separator() +
            user + QDir::separator() + QString(POS_DIR) +
            QDir::separator() + user + QString::number(index) +
            QString(IMAGE_OUTPUT);
        QFile* newImageData = new QFile(filePath);
        while (newImageData->exists()) {
          index ++;
          filePath = QString(FACE_IMAGE_DIR) + QDir::separator() +
              user + QDir::separator() + QString(POS_DIR) +
              QDir::separator() + user + QString::number(index) +
              QString(IMAGE_OUTPUT);
          delete newImageData;
          newImageData = new QFile(filePath);
        }
        delete newImageData;

        // writing image
        cv::imwrite(filePath.toStdString(), face);
        setLog("image written to " + filePath);
      } else {
        // if target is a background image
        // calculate path for bg files
        QString bg = user;
        QString filePath = QString(FACE_IMAGE_DIR) + QDir::separator() +
            bg + QDir::separator() + bg +
            QString::number(index) + QString(IMAGE_OUTPUT);
        QFile* newImageData = new QFile(filePath);
        while (newImageData->exists()) {
          index ++;
          filePath = QString(FACE_IMAGE_DIR) + QDir::separator() +
              bg + QDir::separator() + bg +
              QString::number(index) + QString(IMAGE_OUTPUT);
          delete newImageData;
          newImageData = new QFile(filePath);
        }
        delete newImageData;

        // write the image
        imwrite(filePath.toStdString(), face);
        setLog("image written to " + filePath);
      }
    } else {
      QMessageBox::warning(this, "Warning",
                           "You need to select who the target image is for",
                           QMessageBox::Ok, QMessageBox::NoButton);
      setLog("warning!! you need to select who the target image is for");
    }
    // resume the camera capture
    resume();
  } else {
    QMessageBox::information(this, "Take Picture",
                         "You need to take a picture first",
                         QMessageBox::Ok, QMessageBox::NoButton);
    setLog("warning!! you need to select who the target image is for");
  }
}

void MainWindow::loadNameList() {
  // scan image directory for people list
  ui->selectComboBox->clear();
  ui->selectComboBox->addItem(QString(SELECT));
  QDir imageRoot(FACE_IMAGE_DIR);
  imageRoot.setFilter(QDir::Dirs);
  QStringList dirList = imageRoot.entryList();
  for (int i = 0 ; i < dirList.count() ; i ++) {
    if (dirList[i] != "." && dirList[i] != "..")
      ui->selectComboBox->addItem(dirList[i]);
  }
}

void MainWindow::loadNameMap() {
  // read name mapping xml
  setLog("loading old name mappings...");
  readMap();
  QMapIterator<int, QString> it(names);
  while (it.hasNext()) {
    it.next();
    setLog(QString::number(it.key()) + ": " + it.value());
  }
}

void MainWindow::loadClassifier(QString modelPath) {
    this->faceClassifier->load(modelPath.toStdString());
    setLog("model loaded");
    // set current feature
    FeatureType featureType = faceClassifier->getFeatureType();
    switch (featureType) {
      case classifier::LBP:
        setLog("model uses LBP");
        ui->statusBar->showMessage("current feature: LBP");
        break;
      case classifier::LTP:
        setLog("model uses LTP");
        ui->statusBar->showMessage("current feature: LTP");
        break;
      case classifier::CSLTP:
        setLog("model uses CSLTP");
        ui->statusBar->showMessage("current feature: CSLTP");
        break;
    }
}
