#include "classifier.h"

#define DEFAULT_CLASSIIFIER_TYPE C_SVC
#define DEFAULT_CLASSIIFIER_KERNEL_TYPE RBF
#define DEFAULT_GAMMA 0.1
#define DEFAULT_C 1
#define DEFAULT_NU 0.1
#define DEFAULT_DEGREE 2
#define DEFAULT_COEF0 0.1
#define DEFAULT_P 0

#define LBP_FEATURE_LENGTH 256
#define LTP_FEATURE_LENGTH 9841
#define CSLTP_FEATURE_LENGTH 121

#undef MIN
#define MIN(n1, n2) (n1 < n2 ? n1 : n2)

namespace classifier {

TrainingDataLoader::TrainingDataLoader(const LoadingParams params) {
  this->directory = params.directory;
  this->percent = params.percentForTraining;
  this->featureType = params.featureType;
  this->bgDir = params.bgDir;
  this->posDir = params.posDir;
  this->negDir = params.negDir;
  this->imageSize = params.imageSize;
}

void TrainingDataLoader::load(Mat& trainingData,
                              Mat& trainingLabel,
                              map<int,string>& names) {
  // directory constants
  size_t trainingSize = 0, testingSize = 0;
  vector<string> userFiles, exclusion;
  exclusion.push_back(".");
  exclusion.push_back("..");

  scanDir(directory, userFiles, exclusion);
  for (uint32_t i = 0 ; i < userFiles.size() ; i ++) {
    string path;
    vector<string> imagePaths;
    if (strcmp(userFiles[i].c_str(), bgDir.c_str()) == 0) {
      // background images
      path = directory + string(SEPARATOR) + userFiles[i];
    } else {
      // users images
      path = directory + string(SEPARATOR) + userFiles[i] + posDir;
    }
    scanDir(path, imagePaths, exclusion);
    trainingSize += (size_t) (imagePaths.size() * percent);
    testingSize += (size_t) (imagePaths.size() * (1-percent));

    // mappings
    names.insert(pair<int,string>((i-userFiles.size()/2),
                                  userFiles[i]));
  }

#ifdef DEBUG
  cout << "trainingSize: " << trainingSize << endl;
  cout << "testingSize: " << testingSize << endl;
#endif
  sendMessage(QString("training size: ") +
              QString::number(trainingSize));

  // prepare the matrix for training
  uint32_t featureLength = 0;
  switch (featureType) {
    case LBP:
      featureLength = LBP_FEATURE_LENGTH;
      break;
    case LTP:
      featureLength = LTP_FEATURE_LENGTH;
      break;
    case CSLTP:
      featureLength = CSLTP_FEATURE_LENGTH;
      break;
  }

#ifdef DEBUG
  cout << "feature length: " << featureLength << endl;
#endif

#ifdef QT_DEBUG
  sendMessage(QString("feature length: ") +
              QString::number(featureLength));
#endif

  // data
  Mat tnd = Mat::zeros(trainingSize, featureLength, CV_32FC1);
  Mat ttd = Mat::zeros(testingSize, featureLength, CV_32FC1);
  // labels
  Mat tnl = Mat::zeros(trainingSize, 1, CV_32SC1);
  Mat ttl = Mat::zeros(testingSize, 1, CV_32SC1);
  Mat image, X;

  // extracting lbp/ctlp data for individual samples
  size_t trainingPos = 0, testingPos = 0;
  for (uint32_t i = 0 ; i < userFiles.size() ; i ++) {
    string path;
    vector<string> imagePaths;
    if (strcmp(userFiles[i].c_str(), bgDir.c_str()) == 0) {
      // background images
      path = directory + string(SEPARATOR) + userFiles[i];
    } else {
      // users images
      path = directory + string(SEPARATOR) + userFiles[i] + posDir;
    }
    scanDir(path, imagePaths, exclusion);

#ifdef DEBUG
    cout << "current training size: " << trainingPos << endl;
#endif

#ifdef QT_DEBUG
    sendMessage(QString("current training size: ") +
                QString::number(trainingPos));
#endif

    // read individual images for training
    size_t trainingImageCount = static_cast<size_t>(imagePaths.size() * percent);
    for (uint32_t j = 0 ; j < trainingImageCount ; j ++) {
      QString processingType;
      string briefMat;
      string imagePath = path + string(SEPARATOR) + imagePaths[j];
      image = imread(imagePath);

      if (image.data) {
        Mat resized = Mat::zeros(imageSize, image.type());
        resize(image, resized, imageSize);
        // compute feature
        switch (featureType) {
          case LBP:
            process::computeLBP(resized, X);
            processingType = "LBP";
            break;
          case LTP:
            process::computeLTP(resized, X, 25);
            processingType = "LTP";
            break;
          case CSLTP:
            process::computeCSLTP(resized, X, 25);
            processingType = "CSLTP";
            break;
        }

#ifdef DEBUG
      cout << "tnd.rows = " << tnd.rows << " j = " << j << endl;
#endif

        TrainingDataLoader::brief(X, briefMat);
        sendMessage(QString("Training: loading image from ") +
                    QString(imagePath.c_str()) +
                    QString(" | processing image with ") +
                    processingType);
        sendMessage(QString("Training sample: ") +
                    QString(briefMat.c_str()));

        // copy the x sample to tnd
        for (uint32_t k = 0 ; k < featureLength ; k ++) {
          tnd.ptr<float>()[(j + trainingPos) * tnd.cols + k] =
              X.ptr<float>()[k];
        }

        // set label for this sample
        tnl.ptr<int>()[j + trainingPos] = i - userFiles.size() / 2;
      }
    }
    trainingPos += trainingImageCount;

    // read individual images for testing
    size_t testingImageCount = static_cast<size_t>(imagePaths.size() * (1-percent));
    for (uint32_t j = 0 ; j < testingImageCount ; j ++) {
      QString processingType;
      string imagePath = path + string(SEPARATOR) +
          imagePaths[j + (size_t)(imagePaths.size() * percent)];
      image = imread(imagePath);

      if (image.data) {
        Mat resized = Mat::zeros(imageSize, image.type());
        resize(image, resized, imageSize);
        // compute feature
        switch (featureType) {
          case LBP:
            process::computeLBP(resized, X);
            processingType = "LBP";
            break;
          case LTP:
            process::computeLTP(resized, X, 25);
            processingType = "LTP";
            break;
          case CSLTP:
            process::computeCSLTP(resized, X, 25);
            processingType = "CSLTP";
            break;
        }

        sendMessage(QString("Testing: loading image from ") +
                    QString(imagePath.c_str()) +
                    QString(" | processing image with ") +
                    processingType);

        // copy the x sample to tnd
        for (uint32_t k = 0 ; k < featureLength ; k ++) {
          ttd.ptr<float>()[(j + testingPos) * ttd.cols + k] =
              X.ptr<float>()[k];
        }
        // set label for this sample
        ttl.ptr<int>()[j + testingPos] = i - userFiles.size() / 2;
      }
    }
    testingPos += testingImageCount;
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
  trainingLabel = Mat::zeros(trainingSize + testingSize, 1, CV_32SC1);

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

void TrainingDataLoader::brief(const Mat& mat, string& str) {
  const int maxRows = 10;
  const int maxCols = 20;
  const int rowLimit = MIN(maxRows, mat.rows);
  const int colLimit = MIN(maxCols, mat.cols);

  str = "[";
  if (rowLimit > 0) {
    for (int i = 0 ; i < rowLimit ; i ++) {
      for (int j = 0 ; j < colLimit ; j ++) {
        switch (mat.type()) {
          case CV_8UC3:
            str += std::to_string(
                  mat.ptr<uchar>()[i * mat.cols + j * mat.channels()]);
            break;
          case CV_8UC1:
            str += std::to_string(
                  mat.ptr<uchar>()[i * mat.cols + j]);
            break;
          case CV_32SC1:
            str += std::to_string(
                  mat.ptr<int>()[i * mat.cols + j]);
            break;
          case CV_32FC1:
            str += std::to_string(
                  static_cast<int>(mat.ptr<float>()[i * mat.cols + j]));
            break;
        }
        if (j != colLimit - 1) str += ",";
      }
    }
  }
  str += ", ...]";
}

void loadTrainingData(LoadingParams params,
                      Mat& trainingData,
                      Mat& trainingLabel,
                      map<int,string>& names) {
  // prepare the params
  const string directory = params.directory;
  const string bgDir = params.bgDir;
  const string posDir = params.posDir;
  const double percent = params.percentForTraining;
  const FeatureType featureType = params.featureType;
  const Size size = params.imageSize;

  size_t trainingSize = 0, testingSize = 0;
  vector<string> userFiles, exclusion;
  exclusion.push_back(".");
  exclusion.push_back("..");

  scanDir(directory, userFiles, exclusion);
  for (uint32_t i = 0 ; i < userFiles.size() ; i ++) {
    string path;
    vector<string> imagePaths;
    if (strcmp(userFiles[i].c_str(), bgDir.c_str()) == 0) {
      // background images
      path = directory + string(SEPARATOR) + userFiles[i];
    } else {
      // users images
      path = directory + string(SEPARATOR) + userFiles[i] + posDir;
    }
    scanDir(path, imagePaths, exclusion);
    trainingSize += (size_t) (imagePaths.size() * percent);
    testingSize += (size_t) (imagePaths.size() * (1-percent));

    // mappings
    names.insert(pair<int,string>((i-userFiles.size()/2),
                                  userFiles[i]));
  }

#ifdef DEBUG
  cout << "trainingSize: " << trainingSize << endl;
  cout << "testingSize: " << testingSize << endl;
#endif

  // prepare the matrix for training
  uint32_t featureLength = 0;
  switch (featureType) {
    case LBP:
      featureLength = LBP_FEATURE_LENGTH;
      break;
    case LTP:
      featureLength = LTP_FEATURE_LENGTH;
      break;
    case CSLTP:
      featureLength = CSLTP_FEATURE_LENGTH;
      break;
  }

#ifdef DEBUG
  cout << "feature length: " << featureLength << endl;
#endif

  // data
  Mat tnd = Mat::zeros(trainingSize, featureLength, CV_32FC1);
  Mat ttd = Mat::zeros(testingSize, featureLength, CV_32FC1);
  // labels
  Mat tnl = Mat::zeros(trainingSize, 1, CV_32SC1);
  Mat ttl = Mat::zeros(testingSize, 1, CV_32SC1);
  Mat image, X;

  // extracting feature data for individual samples
  size_t trainingPos = 0, testingPos = 0;
  for (uint32_t i = 0 ; i < userFiles.size() ; i ++) {
    string path;
    vector<string> imagePaths;
    if (strcmp(userFiles[i].c_str(), bgDir.c_str()) == 0) {
      // background images
      path = directory + string(SEPARATOR) + userFiles[i];
    } else {
      // users images
      path = directory + string(SEPARATOR) + userFiles[i] + posDir;
    }
    scanDir(path, imagePaths, exclusion);

#ifdef DEBUG
    cout << "current training size: " << trainingPos << endl;
#endif

    // read individual images for training
    size_t trainingImageCount = static_cast<size_t>(imagePaths.size() * percent);
    for (uint32_t j = 0 ; j < trainingImageCount ; j ++) {
#ifdef DEBUG
      cout << "tnd.rows = " << tnd.rows << " j = " << j << endl;
#endif

      string imagePath = path + string(SEPARATOR) + imagePaths[j];
      image = imread(imagePath);

      if (image.data) {
        Mat resized = Mat::zeros(size, image.type());
        resize(image, resized, size);
        // compute feature
        switch (featureType) {
          case LBP:
            process::computeLBP(resized, X);
            break;
          case LTP:
            process::computeLTP(resized, X, 25);
            break;
          case CSLTP:
            process::computeCSLTP(resized, X, 25);
            break;
        }
        // copy the x sample to tnd
        for (uint32_t k = 0 ; k < featureLength ; k ++) {
          tnd.ptr<float>()[(j + trainingPos) * tnd.cols + k] =
              X.ptr<float>()[k];
        }

        // set label for this sample
        tnl.ptr<int>()[j + trainingPos] = i - userFiles.size() / 2;
      }
    }
    trainingPos += trainingImageCount;

    // read individual images for testing
    size_t testingImageCount = static_cast<size_t>(imagePaths.size() * (1-percent));
    for (uint32_t j = 0 ; j < testingImageCount ; j ++) {
      string imagePath = path + string(SEPARATOR) +
          imagePaths[j + (size_t)(imagePaths.size() * percent)];
      image = imread(imagePath);

      if (image.data) {
        Mat resized = Mat::zeros(size, image.type());
        resize(image, resized, size);
        // compute feature
        switch (featureType) {
          case LBP:
            process::computeLBP(resized, X);
            break;
          case LTP:
            process::computeLTP(resized, X, 25);
            break;
          case CSLTP:
            process::computeCSLTP(resized, X, 25);
            break;
        }

        // copy the x sample to tnd
        for (uint32_t k = 0 ; k < featureLength ; k ++) {
          ttd.ptr<float>()[(j + testingPos) * ttd.cols + k] =
              X.ptr<float>()[k];
        }
        // set label for this sample
        ttl.ptr<int>()[j + testingPos] = i - userFiles.size() / 2;
      }
    }
    testingPos += testingImageCount;

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
  trainingLabel = Mat::zeros(trainingSize + testingSize, 1, CV_32SC1);

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

  this->imageSize = Size(64, 64);

  this->setupSVM();
}

FaceClassifier::FaceClassifier(FaceClassifierParams param) {
  this->type = param.type;
  this->kernelType = param.kernelType;

  this->gamma = param.gamma;
  this->c = param.c;
  this->nu = param.nu;
  this->degree = param.degree;
  this->coef0 = param.coef0;
  this->p = param.p;

  this->imageSize = param.imageSize;

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

#ifdef QT_DEBUG
      sendMessage("Warning!! No such type of SVM implmented");
      sendMessage("Warning!! default to C_SVC");
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

#ifdef QT_DEBUG
      sendMessage("Warning!! No such kernel type of SVM implmented");
      sendMessage("Warning!! default to RBF");
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
                               FaceClassifierKernelType kernelType,
                               Size size) {
  this->type = type;
  this->kernelType = kernelType;

  this->gamma = gamma;
  this->c = c;
  this->nu = nu;
  this->degree = degree;
  this->coef0 = coef0;
  this->p = p;
  this->imageSize = size;

  this->setupSVM();
}

FaceClassifier::FaceClassifier(double gamma, double c, double nu,
                               double degree, double coef0, double p,
                               FaceClassifierType type,
                               FaceClassifierKernelType kernelType,
                               Mat& data, Mat& label,
                               Size size) {
  this->type = type;
  this->kernelType = kernelType;

  this->gamma = gamma;
  this->c = c;
  this->nu = nu;
  this->degree = degree;
  this->coef0 = coef0;
  this->p = p;
  this->imageSize = size;

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

void FaceClassifier::saveModel(string modelPath) {
  this->svm->save(modelPath);
}

void FaceClassifier::train() {
  if (this->trainingData.data && this->trainingLabel.data &&
      this->testingData.data && this->testingLabel.data) {
    // prepare training data
    Ptr<TrainData> td = TrainData::create(trainingData,
                                          ROW_SAMPLE,
                                          trainingLabel);

#ifdef QT_DEBUG
    sendMessage(QString("decreasing training parameter"));
#endif

    double accuracy = 0;
    // cache up original parameters
    this->gammaCache = this->gamma;
    for (unsigned int i = 0 ; i < MAX_ITERATION ; i ++) {
      this->svm->train(td);
      accuracy = this->testAccuracy();

#ifdef DEBUG
      fprintf(stdout, "test accuracy: %lf\n", accuracy);
#endif

      if (accuracy >= TEST_ACCURACY_REQUIREMENT) {
        sendMessage(QString("test accuracy: ") + QString::number(accuracy) +
                    QString(" | requirement reach | stop training ..."));
        return;
      }

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
      sendMessage(QString("test accuracy: ") + QString::number(accuracy) +
                  QString(" | gamma = ") + QString::number(this->gamma) +
                  QString(" | continue to update..."));
    }

#ifdef QT_DEBUG
    sendMessage(QString("increasing training parameter"));
#endif

    this->gamma = this->gammaCache;
    for (unsigned int i = 0 ; i < MAX_ITERATION ; i ++) {
      this->svm->train(td);
      accuracy = this->testAccuracy();

#ifdef DEBUG
      fprintf(stdout, "test accuracy: %lf\n", accuracy);
#endif

      if (accuracy >= TEST_ACCURACY_REQUIREMENT) {
        sendMessage(QString("test accuracy: ") +
                    QString::number(accuracy) +
                    QString(" | requirement reach | stop training ..."));
        return;
      }

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

      sendMessage(QString("test accuracy: ") + QString::number(accuracy) +
                  QString(" | gamma = ") + QString::number(this->gamma) +
                  QString(" | continue to update..."));
    }

    this->gamma = this->gammaCache;
    this->setupSVM();
    this->svm->train(td);

    determineFeatureType();
  } else {
#ifdef DEBUG
    fprintf(stderr, "No training data and label prepared\n");
#endif

#ifdef QT_DEBUG
    sendMessage("no training data and label prepared");
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

#ifdef QT_DEBUG
    sendMessage("input data not suitable for training");
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

#ifdef QT_DEBUG
      sendMessage("SVM not trained");
#endif
      return INT_MAX;
    }
  } else {
#ifdef DEBUG
    fprintf(stderr, "SVM not trained\n");
#endif

#ifdef QT_DEBUG
    sendMessage("SVM not trained");
#endif
    return INT_MAX;
  }
}

int FaceClassifier::predictImageSample(cv::Mat& imageSample) {
  Mat sample, resized;
  string briefMat;

  resized = Mat::zeros(imageSize, imageSample.type());
  resize(imageSample, resized, imageSize);

  switch (this->featureType) {
    case LBP:
      process::computeLBP(resized, sample);
      break;
    case LTP:
      process::computeLTP(resized, sample, 25);
      break;
    case CSLTP:
      process::computeCSLTP(resized, sample, 25);
      break;
  }

  // debug sample matrix
#ifdef DEBUG
  cout << sample << endl;
#endif

#ifdef QT_DEBUG
  TrainingDataLoader::brief(sample, briefMat);
  sendMessage(QString("sample mat: ") + QString(briefMat.c_str()));
#endif

  // if the svm is train then predict the sample
  if (this->svm->isTrained()) {
#ifdef QT_DEBUG
      cout << "feature length: " << this->svm->getVarCount() << endl;
      cout << "sample length: " << sample.cols << endl;
#endif

      return this->svm->predict(sample);
  } else {
#ifdef DEBUG
    fprintf(stderr, "SVM not trained\n");
#endif

#ifdef QT_DEBUG
    sendMessage("SVM not trained");
#endif
    return INT_MAX;
  }
}

void FaceClassifier::load(string modelPath) {
  svm = StatModel::load<SVM>(modelPath);

  determineFeatureType();
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

#ifdef QT_DEBUG
    sendMessage("SVM not trained");
#endif
    return 0;
  }
}

bool FaceClassifier::isLoaded() {
  return svm->isTrained();
}

void FaceClassifier::determineFeatureType() {
  switch (this->svm->getVarCount()) {
    case LBP_FEATURE_LENGTH:
      this->featureType = LBP;
      break;
    case LTP_FEATURE_LENGTH:
      this->featureType = LTP;
      break;
    case CSLTP_FEATURE_LENGTH:
      this->featureType = CSLTP;
      break;
  }
}

FeatureType FaceClassifier::getFeatureType() {
  return this->featureType;
}

} // classifier namespace

// undefine constants
#undef DEFAULT_CLASSIIFIER_TYPE
#undef DEFAULT_CLASSIIFIER_KERNEL_TYPE
#undef DEFAULT_GAMMA
#undef DEFAULT_C
#undef DEFAULT_NU
#undef DEFAULT_DEGREE
#undef DEFAULT_COEF0
#undef DEFAULT_P

#undef LBP_FEATURE_LENGTH
#undef LTP_FEATURE_LENGTH
#undef CSLTP_FEATURE_LENGTH

#undef MIN
