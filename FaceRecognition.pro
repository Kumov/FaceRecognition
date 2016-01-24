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

SOURCES += src/main.cpp\
           src/mainwindow.cpp \
           src/classifier.cpp \
           src/common.cpp \
           src/process.cpp \
           src/opencvcamera.cpp \
           src/imageviewer.cpp \
           src/trainingtask.cpp

HEADERS  += src/mainwindow.h \
            src/classifier.h \
            src/common.h \
            src/process.h \
            src/opencvcamera.h \
            src/imageviewer.h \
            src/trainingtask.h

FORMS    += ui/mainwindow.ui

unix:!macx: LIBS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
                    -lopencv_highgui -lopencv_ml -lopencv_videoio \
                    -lopencv_objdetect -fopenmp
unix:!macx: INCLUDEPATH += /usr/local/include

windows: LIBS += -lopencv_core310 -lopencv_imgproc310 -lopencv_imgcodecs310 \
                 -lopencv_highgui310 -lopencv_ml310 -lopencv_videoio310 \
                 -lopencv_objdetect310
