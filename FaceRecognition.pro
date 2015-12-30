#-------------------------------------------------
#
# Project created by QtCreator 2015-12-27T00:21:13
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = FaceRecognition
TEMPLATE = app

CONFIG += c++11

SOURCES += main.cpp\
        mainwindow.cpp \
    classifier.cpp \
    common.cpp \
    process.cpp \
    opencvcamera.cpp \
    imageviewer.cpp \
    trainingtask.cpp

HEADERS  += mainwindow.h \
    classifier.h \
    common.h \
    process.h \
    opencvcamera.h \
    imageviewer.h \
    trainingtask.h

FORMS    += mainwindow.ui

unix:!macx: LIBS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_ml -lopencv_videoio -lopencv_objdetect -fopenmp
unix:!macx: INCLUDEPATH += /usr/local/include
