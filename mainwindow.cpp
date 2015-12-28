#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent),
                                          ui(new Ui::MainWindow),
                                          timer(new QTimer()) {
  ui->setupUi(this);

  imageViewer = new ImageViewer();
  ui->mainWidget->layout()->addWidget(imageViewer);

  timer->start(INTERVAL);
  connect(timer, SIGNAL(timeout()), this, SLOT(setImage()));
}

MainWindow::~MainWindow() {
  if (timer != nullptr)
    delete timer;
  if (ui != nullptr)
    delete ui;
}

void MainWindow::setImage() {
  QImage image = camera.getCurrentFrame();
#ifdef QT_DEBUG
  cout << "getting image" << endl;
#endif
  imageViewer->setImage(image);
}
