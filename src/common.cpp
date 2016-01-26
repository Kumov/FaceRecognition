#include "common.h"

// constants
const char* const DEFAULT_BG_DIR = "bg";
const char* const DEFAULT_POS_DIR = "/pos";
const char* const DEFAULT_NEG_DIR = "/pos";
const char* const DEFAULT_MODEL_OUTPUT = "facemodel.xml";

static bool contain(const vector<string> data, const char* s) {
  for (uint32_t i = 0 ; i < data.size(); i ++) {
    if (strcmp(s, data[i].c_str()) == 0)
      return true;
  }
  return false;
}

vector<string> scanDir(const string path, const vector<string> exclusion) {
  vector<string> files;

#if defined(__unix__)
  DIR *dir = NULL;
  struct dirent *entity = NULL;
  if((dir = opendir(path.c_str())) != NULL) {
    while((entity = readdir(dir)) != NULL) {
      // exclude . , .. , output file, and vector file
      if(!contain(exclusion, entity->d_name)) {
        const string file(entity->d_name);
        files.push_back(file);
      }
    }
    closedir(dir);
  } else {
#ifdef DEBUG
    fprintf(stderr, "Fail to open %s\n", path.c_str());
#endif
  }
#elif defined(__WIN32)
  DIR *dir = NULL;
  struct dirent *entity = NULL;
  if((dir = opendir(path.c_str())) != NULL) {
    while((entity = readdir(dir)) != NULL) {
      // exclude . , .. , output file, and vector file
      if(!contain(exclusion, entity->d_name)) {
        const string file(entity->d_name);
        files.push_back(file);
      }
    }
    closedir(dir);
  } else {
#ifdef DEBUG
    fprintf(stderr, "Fail to open %s\n", path.c_str());
#endif
  }
#endif
  return files;
}

void scanDir(const string path, vector<string>& files,
    const vector<string> exclusion) {
#if defined(__unix__)
  vector<string> scanResult;
  DIR *dir = NULL;
  struct dirent *entity = NULL;
  if((dir = opendir(path.c_str())) != NULL) {
    while((entity = readdir(dir)) != NULL) {
      // exclude . , .. , output file, and vector file
      if(!contain(exclusion, entity->d_name)) {
        const string file(entity->d_name);
        scanResult.push_back(file);
      }
    }
    closedir(dir);
    for (uint32_t i = 0 ; i < scanResult.size() ; i ++) {
      if (scanResult[i] != string(DEFAULT_BG_DIR)) {
        files.push_back(scanResult[i]);
      }
    }
    files.push_back(string(DEFAULT_BG_DIR));
  } else {
#ifdef DEBUG
    fprintf(stderr, "Fail to open %s\n", path.c_str());
#endif
  }
#elif defined(__WIN32)
    vector<string> scanResult;
    DIR *dir = NULL;
    struct dirent *entity = NULL;
    if((dir = opendir(path.c_str())) != NULL) {
      while((entity = readdir(dir)) != NULL) {
        // exclude . , .. , output file, and vector file
        if(!contain(exclusion, entity->d_name)) {
          const string file(entity->d_name);
          scanResult.push_back(file);
        }
      }
      closedir(dir);
      for (uint32_t i = 0 ; i < scanResult.size() ; i ++) {
        if (scanResult[i] != string(DEFAULT_BG_DIR)) {
          files.push_back(scanResult[i]);
        }
      }
      files.push_back(string(DEFAULT_BG_DIR));
    } else {
  #ifdef DEBUG
      fprintf(stderr, "Fail to open %s\n", path.c_str());
  #endif
    }
#endif
}

bool fileExists(const string filePath) {
  FILE *file = NULL;
  if ((file = fopen(filePath.c_str(), "r"))) {
    fclose(file);
    return true;
  }
  return false;
}

uint32_t getLineCount(const string filePath) {
  uint32_t count = 0;
  ifstream file;
  file.open(filePath.c_str(), ios::in);

  for (string line; getline(file, line) ;) {
    count ++;
  }
  file.close();

  return count;
}

bool createDirectory(const string name) {
#if defined(__unix__)
  if (mkdir(name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
    return false;
  else
    return true;
#elif defined(__WIN32)
  size_t pathLength = sizeof(wchar_t) * (name.length() + 1);
  wchar_t* longFilePath = (wchar_t*) malloc(pathLength);
  memset(longFilePath, L'\0', pathLength);
  swprintf(longFilePath, L"%hs", name.c_str());
  if (CreateDirectory(longFilePath, NULL) == 0) {
      return false;
  }
  return true;
#endif
}

bool deleteFile(const string filePath) {
#if defined(__unix__)
  int status = remove(filePath.c_str());
  if (status == 0) {
    return true;
  }
  else if (status == -1) {
    return false;
  }
#elif defined(__WIN32)
  size_t pathLength = sizeof(wchar_t) * (filePath.length() + 1);
  wchar_t* longFilePath = (wchar_t*) malloc(pathLength);
  memset(longFilePath, L'\0', pathLength);
  swprintf(longFilePath, L"%hs", filePath.c_str());
  DeleteFile(longFilePath);
#endif
  return true;
}
