#include "imageviewer.h"

void ImageViewer::setImage(QImage img) {
  image = img;
  repaint();
}

void ImageViewer::paintEvent(QPaintEvent *) {
  QPainter painter(this);

  if (!this->image.isNull()) {
    const int windowWidth = this->width();
    const int windowHeight = this->height();
    const double windowRatio = static_cast<double>(windowWidth) /
        static_cast<double>(windowHeight);
    const int imageWidth = image.width();
    const int imageHeight = image.height();
    const double imageRatio = static_cast<double>(imageWidth) /
        static_cast<double>(imageHeight);
    if (windowRatio > imageRatio) {
      const int imageNewHeight = windowHeight;
      const int imageNewWidth = static_cast<int>(imageRatio *
                                                 imageNewHeight);
      QImage scaled = image.scaled(imageNewWidth, imageNewHeight);
      QPoint start = QPoint((windowWidth - imageNewWidth) / 2, 0);
      painter.drawImage(start, scaled);
    } else {
      const int imageNewWidth = windowWidth;
      const int imageNewHeight = static_cast<int>(imageNewWidth /
                                                  imageRatio);
      QImage scaled = image.scaled(imageNewWidth, imageNewHeight);
      QPoint start = QPoint(0, (windowHeight - imageNewHeight) / 2);
      painter.drawImage(start, scaled);
    }
  }
}
