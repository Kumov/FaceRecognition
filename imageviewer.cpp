#include "imageviewer.h"

void ImageViewer::setImage(QImage img) {
  image = img;
  repaint();
}

void ImageViewer::paintEvent(QPaintEvent *) {
   QPainter painter(this);
   painter.drawImage(QPoint(0, 0), image);
}
