#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <QtCore>
#include <QObject>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <limits.h>
#include <map>
#include <vector>

#include "process.h"
#include "common.h"

#ifdef QT_DEBUG
using std::cout;
using std::endl;
#endif

using std::string;
using std::map;
using std::vector;
using std::pair;
using cv::Mat;
using cv::ml::TrainData;
using cv::ml::ROW_SAMPLE;
using cv::ml::SVM;
using cv::Ptr;
using cv::imread;
using cv::ml::StatModel;

namespace classifier {
// supported feature type
typedef enum {
  LBP, CTLP
} FeatureType;

// params for loading
typedef struct LoadingParams {
  LoadingParams() {}
  LoadingParams(string dir, double percent, FeatureType type) {
    directory = dir;
    if (percent <= 1.0 && percent >= 0) {
      percentForTraining = percent;
    } else {
      percentForTraining = 1.0;
    }
    featureType = type;
  }
  string directory;
  double percentForTraining;
  FeatureType featureType;
} LoadingParams;

// loading functions
void loadTrainingData(LoadingParams params,
                      Mat& trainingData,
                      Mat& trainingLabel,
                      map<int,string>& names);

struct FaceClassifierParams;

class FaceClassifier : public QObject {
  Q_OBJECT
 public:
  // constants
  enum FaceClassifierType {
    C_SVC,
    // C-Support Vector Classification.
    // n-class classification (n \geq 2),
    // allows imperfect separation of classes
    // with penalty multiplier C for outliers.
    NU_SVC,
    // nu-Support Vector Classification.
    // n-class classification with possible
    // imperfect separation. Parameter nu
    // (in the range 0..1, the larger the value,
    // the smoother the decision boundary)
    // is used instead of C.
    ONE_CLASS,
    // Distribution Estimation (One-class SVM).
    // All the training data are from the same class,
    // SVM builds a boundary that separates the class
    // from the rest of the feature space.
    EPS_SVR,
    // epsilon-Support Vector Regression.
    // The distance between feature vectors from
    // the training set and the fitting hyper-plane
    // must be less than p. For outliers the penalty
    // multiplier C is used.
    NU_SVR
    // nu-Support Vector Regression.
    // nu is used instead of p.
  };

  enum FaceClassifierKernelType {
    LINEAR,
    // Linear kernel.
    // No mapping is done, linear discrimination
    // (or regression) is done in the original
    // feature space. It is the fastest option.
    // K(x_i, x_j) = x_i^T x_j.
    POLY,
    // Polynomial kernel:
    // K(x_i, x_j) = (gamma x_i^T x_j + coef0)^{degree},
    // gamma > 0.
    RBF,
    // Radial basis function (RBF),
    // a good choice in most cases.
    // K(x_i, x_j) = e^{-gamma ||x_i - x_j||^2}, gamma > 0.
    SIGMOID
    // Sigmoid kernel:
    // K(x_i, x_j) = \tanh(\gamma x_i^T x_j + coef0).
  };

  FaceClassifier();
  explicit FaceClassifier(struct FaceClassifierParams param);
  FaceClassifier(double gamma, double c, double nu,
                 double degree, double coef0, double p,
                 FaceClassifierType type,
                 FaceClassifierKernelType kernelType);
  FaceClassifier(double gamma, double c, double nu,
                 double degree, double coef0, double p,
                 FaceClassifierType type,
                 FaceClassifierKernelType kernelType,
                 Mat& data, Mat& label);
  virtual ~FaceClassifier() {}
  void saveModel();
  void saveModel(string modelPath);
  void train();
  void train(Mat& data, Mat& label);
  int predict(Mat& sample);
  int predictImageSample(Mat& imageSample);
  void load(string modelPath);
  double testAccuracy();
  bool isLoaded();

 signals:
  void sendMessage(string message);

 protected:
  void setupSVM();

 private:
  Ptr<SVM> svm;
  FaceClassifierType type;
  FaceClassifierKernelType kernelType;
  FeatureType featureType = LBP;
  double gamma, c, nu, degree, coef0, p;
  double gammaCache;
  Mat trainingData, testingData;
  Mat trainingLabel, testingLabel;
  const string MODEL_OUTPUT = "facemodel.xml";
  const double TEST_ACCURACY_REQUIREMENT = 0.96;
  const double TEST_PERCENT = 0.1;
  const unsigned long long int MAX_ITERATION = 100;
};

typedef struct FaceClassifierParams {
  double gamma;
  double c;
  double nu;
  double degree;
  double coef0;
  double p;
  FaceClassifier::FaceClassifierType type;
  FaceClassifier::FaceClassifierKernelType kernelType;

  FaceClassifierParams() {
    gamma = 1.0;
    c = 1.0;
    nu = 1.0;
    degree = 1.0;
    coef0 = 0;
    p = 0;
    type = FaceClassifier::C_SVC;
    kernelType = FaceClassifier::LINEAR;
  }

  FaceClassifierParams(double _gamma) {
    gamma = _gamma;
    c = 1.0;
    nu = 1.0;
    degree = 1.0;
    coef0 = 0;
    p = 0;
    type = FaceClassifier::C_SVC;
    kernelType = FaceClassifier::RBF;
  }
} FaceClassifierParams;

} /* classifier */


#endif /* end of include guard: CLASSIFIER_H */
