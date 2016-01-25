### Face Recognition Project
[![Travis Build Status](https://travis-ci.org/kiddos/FaceRecognition.svg?branch=master)](https://travis-ci.org/kiddos/FaceRecognition)

This project is aimed to differentiate face from different people by using basic svm classification and methods of feature extraction

library used:
- Qt 5.5.x
- opencv 3.0.0

note:
After couple attemps, I found that the implemented SVM (type C_SVC) was train starting from the lowest value label. In such case, my background image should be placed with the highest label value, so that if input image cannot be classify into previous cases (users) that are classify as background image (does not belong to the user group)
