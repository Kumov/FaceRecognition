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

#define DEFAULT_BG_DIR "bg"
#define DEFAULT_POS_DIR "/pos"
#define DEFAULT_NEG_DIR "/neg"
#define DEFAULT_TEST_PERCENT 0.1
#define ACCURACY_REQUIREMENT 0.966
#define DEFAULT_TRAINING_STEP 0.05
#define ITERATION 1000

#ifdef QT_DEBUG
using std::cout;
using std::endl;
#endif

using std::string;
using std::map;
using std::vector;
using std::pair;
using cv::Mat;
using cv::Size;
using cv::ml::TrainData;
using cv::ml::ROW_SAMPLE;
using cv::ml::SVM;
using cv::Ptr;
using cv::imread;
using cv::resize;
using cv::ml::StatModel;

namespace classifier {
// supported feature type
typedef enum {
  LBP,      // local binary pattern
  LTP,      // local ternary pattern
  CSLTP     // central symmetric local ternary pattern
} FeatureType;

// params for loading
typedef struct LoadingParams {
  LoadingParams() {}
  LoadingParams(string dir, double percent,
                FeatureType type,
                Size size) {
    directory = dir;
    if (percent <= 1.0 && percent >= 0) {
      percentForTraining = percent;
    } else {
      percentForTraining = 1.0;
    }
    featureType = type;
    imageSize = size;
  }

  LoadingParams(string dir, string bg,
                string pos, string neg,
                double percent, FeatureType type,
                Size size) {
    directory = dir;
    if (percent <= 1.0 && percent >= 0) {
      percentForTraining = percent;
    } else {
      percentForTraining = 1.0;
    }
    featureType = type;
    bgDir = bg;
    posDir = pos;
    negDir = neg;
    imageSize = size;
  }

  string directory;
  string bgDir = DEFAULT_BG_DIR;
  string posDir = DEFAULT_POS_DIR;
  string negDir = DEFAULT_NEG_DIR;
  double percentForTraining;
  FeatureType featureType;
  Size imageSize;
} LoadingParams;

class TrainingDataLoader : public QObject {
  Q_OBJECT
 public:
  TrainingDataLoader(const LoadingParams params);
  virtual ~TrainingDataLoader() {}
  void load(Mat& trainingData, Mat& trainingLabel,
       map<int, string>& names);
  static void brief(const Mat& mat, string& str);

 signals:
  void sendMessage(QString message);

 private:
  string directory;
  string bgDir, posDir, negDir;
  double percent;
  FeatureType featureType;
  Size imageSize;
};

// old function for loading training data
// should use the TrainingDataLoader class and
// specify LoadingParams for loading data
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
  FaceClassifier(struct FaceClassifierParams param,
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
  void determineFeatureType();
  FeatureType getFeatureType();

 signals:
  void sendMessage(QString message);

 protected:
  void setupSVM();
  void setupTrainingData(Mat& data, Mat& label);

 private:
  Ptr<SVM> svm;
  FaceClassifierType type;
  FaceClassifierKernelType kernelType;
  FeatureType featureType;
  double gamma, c, nu, degree, coef0, p;
  double gammaCache;
  double trainingStep, testPercent;
  Mat trainingData, testingData;
  Mat trainingLabel, testingLabel;
  Size imageSize;
  const string MODEL_OUTPUT = "facemodel.xml";
  const double TEST_ACCURACY_REQUIREMENT = ACCURACY_REQUIREMENT;
  const unsigned long long int MAX_ITERATION = ITERATION;
};

typedef struct FaceClassifierParams {
  double gamma;
  double c;
  double nu;
  double degree;
  double coef0;
  double p;
  double trainingStep;
  FaceClassifier::FaceClassifierType type;
  FaceClassifier::FaceClassifierKernelType kernelType;
  Size imageSize;
  double testingPercent;

  FaceClassifierParams() {
    gamma = 1.0;
    c = 1.0;
    nu = 1.0;
    degree = 1.0;
    coef0 = 0;
    p = 0;
    type = FaceClassifier::C_SVC;
    kernelType = FaceClassifier::LINEAR;
    trainingStep = DEFAULT_TRAINING_STEP;
    testingPercent = DEFAULT_TEST_PERCENT;
  }

  FaceClassifierParams(Size size,
                       double _gamma,
                       double _trainingStep,
                       double _testingPercent) {
    gamma = _gamma;
    c = 1.0;
    nu = 1.0;
    degree = 1.0;
    coef0 = 0;
    p = 0;
    type = FaceClassifier::C_SVC;
    kernelType = FaceClassifier::RBF;
    imageSize = size;
    trainingStep = _trainingStep;
    if (_testingPercent < 1.0 && _testingPercent > 0) {
      testingPercent = _testingPercent;
    } else {
      testingPercent = DEFAULT_TEST_PERCENT;
    }
  }
} FaceClassifierParams;

} /* classifier */

#undef DEFAULT_BG_DIR
#undef DEFAULT_POS_DIR
#undef DEFAULT_NEG_DIR

#endif /* end of include guard: CLASSIFIER_H */
