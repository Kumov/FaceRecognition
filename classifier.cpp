#include "classifier.h"

#define DEFAULT_CLASSIIFIER_TYPE C_SVC
#define DEFAULT_CLASSIIFIER_KERNEL_TYPE RBF
#define DEFAULT_GAMMA 0.1
#define DEFAULT_C 1
#define DEFAULT_NU 0.1
#define DEFAULT_DEGREE 2
#define DEFAULT_COEF0 0.1
#define DEFAULT_P 0

#define BG_DIR "bg"
#define POS_DIR "/pos"
#define NEG_DIR "/neg"

namespace classifier {

void loadTrainingData(LoadingParams params,
                      Mat& trainingData,
                      Mat& trainingLabel,
                      map<int,string>& names) {
  // prepare the params
  const string directory = params.directory;
  const double percent = params.percentForTraining;
  const FeatureType featureType = params.featureType;

  size_t trainingSize = 0, testingSize = 0;
  Mat tnd, tnl, ttd, ttl, X;
  vector<vector<Mat> > allImages;
  vector<string> userFiles, exclusion;
  exclusion.push_back(".");
  exclusion.push_back("..");

  scanDir(directory, userFiles, exclusion);

  // read all images to vectors allImages
  for (uint32_t i = 0 ; i < userFiles.size() ; i ++) {
    string path;
    vector<Mat> images;
    vector<string> imagePaths;

    // there is no positive or negative background, just background
    if (strcmp(userFiles[i].c_str(), BG_DIR) == 0) {
      path = directory + string(SEPARATOR) + userFiles[i];
    } else {
      path = directory + string(SEPARATOR) + userFiles[i] + string(POS_DIR);
    }
    scanDir(path, imagePaths, exclusion);

    // read individual images
    for (uint32_t j = 0 ; j < imagePaths.size() ; j ++) {
      string imagePath = path + string(SEPARATOR) + imagePaths[i];
      Mat image = imread(imagePath);
      if (image.data) {
        images.push_back(image);
      }
    }
    // add to allImages
    allImages.push_back(images);
    // mappings
    names.insert(pair<int,string>((i-userFiles.size()/2), userFiles[i]));
  }

  // set the training/testingSize according to the specified percent
  for (uint32_t i = 0 ; i < allImages.size() ; i ++) {
    trainingSize += (size_t) ((double)allImages[i].size() * percent);
    testingSize += (size_t) ((double)allImages[i].size() * (1 - percent));
  }

  // debug info
#ifdef DEBUG
  cout << "trainingSize: " << trainingSize << endl;
  cout << "testingSize: " << testingSize << endl;
#endif

  // prepare the matrix for training
  uint32_t featureLength = 0;
  switch (featureType) {
    case LBP:
      featureLength = 256;
      break;
    case CTLP:
      featureLength = 256;
      break;
  }
  // data
  tnd = Mat::zeros(trainingSize, featureLength, CV_32FC1);
  ttd = Mat::zeros(testingSize, featureLength, CV_32FC1);
  // labels
  tnl = Mat::zeros(trainingSize, 1, CV_32SC1);
  ttl = Mat::zeros(testingSize, 1, CV_32SC1);

  // extracting lbp/ctlp data for individual samples
  X = Mat(1, featureLength, CV_32SC1);
  size_t trainingPos = 0, testingPos = 0;
  for (uint32_t i = 0 ; i < allImages.size() ; i ++) {
    vector<Mat> images = allImages[i];

    // training data/labels: tnd tnl
    size_t tns = (size_t)((double)images.size() * percent);
    for (uint32_t j = 0 ; j < tns ; j ++) {
      Mat image = images[j];
      float* rowData = tnd.ptr<float>() + (j + trainingPos) * tnd.cols;

      switch (featureType) {
        case LBP:
          process::computeLBP(image, X);
          break;
        case CTLP:
          // TODO
          // implement CTLP for processing module
          break;
      }

      // copy the X sample to tnd
      for (int32_t k = 0 ; k < tnd.cols ; k ++) {
        rowData[k] = X.ptr<int>()[k];
      }
      // set label for this sample
      tnl.ptr<int>()[j + trainingPos] = i - allImages.size() / 2;
    }

#ifdef DEBUG
    cout << "current training size: " << trainingData << endl;
    cout << "current training size: " << trainingLabel << endl;
#endif
    trainingPos += tns;

    // testing data/labels: ttd ttl
    size_t tts = (size_t)((double)images.size() * (1 - percent));
    for (uint32_t j = 0 ; j < tts ; j ++) {
      Mat image = images[j + tns];
      float* rowData = ttd.ptr<float>() + (j + testingPos) * ttd.cols;

      switch (featureType) {
        case LBP:
          process::computeLBP(image, X);
          break;
        case CTLP:
          // TODO
          // implement CTLP for processing module
          break;
      }

      // copy the X sample to tnd
      for (int32_t k = 0 ; k < ttd.cols ; k ++) {
        rowData[k] = X.ptr<int>()[k];
      }
      // set label for this sample
      ttl.ptr<int>()[j + testingPos] = i - allImages.size() / 2;
    }

#ifdef DEBUG
    cout << "current testing size: " << testingSize << endl;
#endif
    testingPos += tts;
  }

  // debug info
#ifdef DEBUG
  cout << tnd << endl;
  cout << ttd << endl;
  cout << tnl << endl;
  cout << ttl << endl;
#endif

  // put all data/lables into trainingData/Labels
  trainingData = Mat::zeros(trainingSize + testingSize, featureLength, CV_32FC1);
  trainingLabel = Mat::zeros(trainingSize + testingSize, featureLength, CV_32FC1);

  const uint32_t cols = trainingData.cols;
  for (uint32_t i = 0 ; i < trainingSize ; i ++) {
    // training data
    float* rowData = tnd.ptr<float>() + i * tnd.cols;
    for (uint32_t j = 0 ; j < cols ; j ++) {
      trainingData.ptr<float>()[i * cols + j] = rowData[j];
    }
    // training label
    trainingLabel.ptr<int>()[i] = tnl.ptr<int>()[i];
  }
  for (uint32_t i = 0 ; i < testingSize ; i ++) {
    // testing data
    float* rowData = ttd.ptr<float>() + i * ttd.cols;
    for (uint32_t j = 0 ; j < cols ; j ++) {
      trainingData.ptr<float>()[(i + trainingSize) * cols + j] = rowData[j];
    }
    // testing label
    trainingLabel.ptr<int>()[i + trainingSize] = ttl.ptr<int>()[i];
  }

#ifdef DEBUG
  cout << trainingData << endl;
  cout << trainingLabel << endl;
#endif
}

FaceClassifier::FaceClassifier() {
  this->type = DEFAULT_CLASSIIFIER_TYPE;
  this->kernelType = DEFAULT_CLASSIIFIER_KERNEL_TYPE;

  this->gamma = DEFAULT_GAMMA;
  this->c = DEFAULT_C;
  this->nu = DEFAULT_NU;
  this->degree = DEFAULT_DEGREE;
  this->coef0 = DEFAULT_COEF0;
  this->p = DEFAULT_P;

  this->setupSVM();
}

void FaceClassifier::setupSVM() {
  this->svm = SVM::create();

  switch (this->type) {
    case C_SVC:
      this->svm->setType(SVM::C_SVC);
      break;
    case NU_SVC:
      this->svm->setType(SVM::NU_SVC);
      break;
    case ONE_CLASS:
      this->svm->setType(SVM::ONE_CLASS);
      break;
    case EPS_SVR:
      this->svm->setType(SVM::EPS_SVR);
      break;
    case NU_SVR:
      this->svm->setType(SVM::NU_SVR);
      break;
    default:
#ifdef DEBUG
      fprintf(stderr, "No such type of SVM implemented\n");
      fprintf(stderr, "default to C_SVC\n");
#endif
      this->svm->setType(cv::ml::SVM::C_SVC);
      break;
  }

  switch (this->kernelType) {
    case LINEAR:
      this->svm->setKernel(SVM::LINEAR);
      break;
    case POLY:
      this->svm->setKernel(SVM::POLY);
      break;
    case RBF:
      this->svm->setKernel(SVM::RBF);
      break;
    case SIGMOID:
      this->svm->setKernel(SVM::SIGMOID);
      break;
    default:
#ifdef DEBUG
      fprintf(stderr, "No such kernel type of SVM implemented\n");
      fprintf(stderr, "default to RBF\n");
#endif
      this->svm->setKernel(SVM::RBF);
      break;
  }

  this->svm->setC(this->c);
  this->svm->setNu(this->nu);
  this->svm->setGamma(this->gamma);
  this->svm->setCoef0(this->coef0);
  this->svm->setP(this->p);
}

FaceClassifier::FaceClassifier(double gamma, double c, double nu,
                               double degree, double coef0, double p,
                               FaceClassifierType type,
                               FaceClassifierKernelType kernelType) {
  this->type = type;
  this->kernelType = kernelType;

  this->gamma = gamma;
  this->c = c;
  this->nu = nu;
  this->degree = degree;
  this->coef0 = coef0;
  this->p = p;

  this->setupSVM();
}

FaceClassifier::FaceClassifier(double gamma, double c, double nu,
                               double degree, double coef0, double p,
                               FaceClassifierType type,
                               FaceClassifierKernelType kernelType,
                               Mat& data, Mat& label) {
  this->type = type;
  this->kernelType = kernelType;

  this->gamma = gamma;
  this->c = c;
  this->nu = nu;
  this->degree = degree;
  this->coef0 = coef0;
  this->p = p;

  this->setupSVM();

  size_t testingSize = data.rows * TEST_PERCENT > 1 ? data.rows * TEST_PERCENT : 1;
  size_t trainingSize = data.rows - testingSize;

  if (data.type() == CV_32FC1 && label.type() == CV_32SC1) {
    trainingData = cv::Mat(trainingSize, data.cols, data.type());
    testingData = cv::Mat(testingSize, data.cols, data.type());
    trainingLabel = cv::Mat(trainingSize, label.cols, label.type());
    testingLabel = cv::Mat(testingSize, label.cols, label.type());

    for (int i = 0 ; i < trainingData.rows ; i ++) {
      for (int j = 0 ; j < trainingData.cols ; j ++) {
        trainingData.ptr<float>()[i*trainingData.cols + j] =
          data.ptr<float>()[i*trainingData.cols + j];
      }
      trainingLabel.ptr<int>()[i] = label.ptr<int>()[i];
    }
    size_t doffset = trainingData.rows * trainingData.cols;
    size_t loffset = trainingLabel.rows;
    for (int i = 0 ; i < testingData.rows ; i ++) {
      for (int j = 0 ; j < testingData.cols ; j ++) {
        testingData.ptr<float>()[i*testingData.cols + j] =
          data.ptr<float>()[doffset + i*testingData.cols + j];
      }
      testingLabel.ptr<int>()[i] = label.ptr<int>()[loffset + i];
    }
  }
}

void FaceClassifier::saveModel() {
	this->svm->save(MODEL_OUTPUT);
}

void FaceClassifier::train() {
  if (this->trainingData.data && this->trainingLabel.data &&
      this->testingData.data && this->testingLabel.data) {
    // prepare training data
    Ptr<TrainData> td = TrainData::create(trainingData, ROW_SAMPLE, trainingLabel);

    // cache up original parameters
    this->gammaCache = this->gamma;

    double accuracy = 0;
    for (unsigned int i = 0 ; i < MAX_ITERATION ; i ++) {
      this->svm->train(td);
      accuracy = this->testAccuracy();

#ifdef DEBUG
      fprintf(stdout, "test accuracy: %lf\n", accuracy);
#endif

      if (accuracy >= TEST_ACCURACY_REQUIREMENT) return;

      double l = 0;
      switch (this->kernelType) {
        case LINEAR:
          break;
        case POLY:
          l = log10(this->gamma);
          l += 0.1;
          this->gamma = pow(10, l);
          break;
        case RBF:
          l = log10(this->gamma);
          l += 0.1;
          this->gamma = pow(10, l);
          break;
        case SIGMOID:
          l = log10(this->gamma);
          l += 0.1;
          this->gamma = pow(10, l);
          break;
      }
      this->setupSVM();
    }

    this->gamma = this->gammaCache;
    for (unsigned int i = 0 ; i < MAX_ITERATION ; i ++) {
      this->svm->train(td);
      accuracy = this->testAccuracy();
#ifdef DEBUG
      fprintf(stdout, "test accuracy: %lf\n", accuracy);
#endif

      if (accuracy >= TEST_ACCURACY_REQUIREMENT)
        return;

      double l = 0;
      switch (this->kernelType) {
        case LINEAR:
          break;
        case POLY:
          l = log10(this->gamma);
          l -= 0.1;
          this->gamma = pow(10, l);
          break;
        case RBF:
          l = log10(this->gamma);
          l -= 0.1;
          this->gamma = pow(10, l);
          break;
        case SIGMOID:
          l = log10(this->gamma);
          l -= 0.1;
          this->gamma = pow(10, l);
          break;
      }
      this->setupSVM();
    }

    this->gamma = this->gammaCache;
    this->setupSVM();
    this->svm->train(td);
  } else {
#ifdef DEBUG
    fprintf(stderr, "No training data and label prepared\n");
#endif
  }
}

void FaceClassifier::train(Mat& data, Mat& label) {
  size_t testingSize = data.rows * TEST_PERCENT > 1 ? data.rows * TEST_PERCENT : 1;
  size_t trainingSize = data.rows - testingSize;

  if (data.data && data.type() == CV_32FC1 &&
      label.data && label.type() == CV_32SC1) {
    trainingData = Mat(trainingSize, data.cols, data.type());
    testingData = Mat(testingSize, data.cols, data.type());
    trainingLabel = Mat(trainingSize, label.cols, label.type());
    testingLabel = Mat(testingSize, label.cols, label.type());

    for (int i = 0 ; i < trainingData.rows ; i ++) {
      for (int j = 0 ; j < trainingData.cols ; j ++) {
        trainingData.ptr<float>()[i*trainingData.cols + j] =
          data.ptr<float>()[i*trainingData.cols + j];
      }
      trainingLabel.ptr<int>()[i] = label.ptr<int>()[i];
    }

    size_t doffset = trainingData.rows * trainingData.cols;
    size_t loffset = trainingLabel.rows;
    for (int i = 0 ; i < testingData.rows ; i ++) {
      for (int j = 0 ; j < testingData.cols ; j ++) {
        testingData.ptr<float>()[i*testingData.cols + j] =
          data.ptr<float>()[doffset + i*testingData.cols + j];
      }
      testingLabel.ptr<int>()[i] = label.ptr<int>()[loffset + i];
    }
  } else {
#ifdef DEBUG
    fprintf(stderr, "input data not suitable for training\n");
#endif
  }
  this->train();
}

int FaceClassifier::predict(cv::Mat& sample) {
  if (this->svm->isTrained()) {
    if (sample.rows == 1 && sample.cols == trainingData.cols &&
        sample.type() == trainingData.type()) {
      return this->svm->predict(sample);
    } else {
#ifdef DEBUG
      fprintf(stderr, "SVM not trained\n");
#endif
      return INT_MAX;
    }
  } else {
#ifdef DEBUG
    fprintf(stderr, "SVM not trained\n");
#endif
    return INT_MAX;
  }
}

double FaceClassifier::testAccuracy() {
  if (this->svm->isTrained()) {
    size_t correct = 0;
    Mat testResult;
    this->svm->predict(testingData, testResult);

    for (int i = 0 ; i < testResult.rows ; i ++) {
      if (testResult.ptr<float>()[i] == testingLabel.ptr<int>()[i])
        correct ++;
    }
    return (double) correct / testingLabel.rows;
  } else {
#ifdef DEBUG
    fprintf(stderr, "SVM not trained\n");
#endif
    return 0;
  }
}
}

// undefine constants
#undef DEFAULT_CLASSIIFIER_TYPE
#undef DEFAULT_CLASSIIFIER_KERNEL_TYPE
#undef DEFAULT_GAMMA
#undef DEFAULT_C
#undef DEFAULT_NU
#undef DEFAULT_DEGREE
#undef DEFAULT_COEF0
#undef DEFAULT_P
#undef POS_DIR
#undef NEG_DIR
