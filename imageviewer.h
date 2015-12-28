#ifndef IMAGEVIEWER_H
#define IMAGEVIEWER_H

#include <QWidget>
#include <QPaintEvent>
#include <QPainter>

class ImageViewer : public QWidget {
  Q_OBJECT
 public:
  explicit ImageViewer(QWidget *parent = 0) : QWidget(parent) {}
  void setImage(QImage image);
  virtual void paintEvent(QPaintEvent *);

 private:
  QImage image;
};

#endif // IMAGEVIEWER_H
