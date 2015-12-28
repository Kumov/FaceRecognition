#ifndef COMMON_H
#define COMMON_H

#include <opencv2/core/core.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <cmath>

#if defined(__unix__)
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#elif defined(__WIN32)
// code for windows
#include <windows.h>
#endif

#define   CURRENT_DIR   "."
#define   PARENT_DIR    ".."

// separator
#if defined(__unix__)
#define   SEPARATOR     "/"
#elif defined(__WIN32)
// windows implementation ...
#define   SEPARATOR     "\\"
#endif

// premade classifier data
#if defined(__unix__)
#define CLASSIFIER_DATA	\
	"/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml"
#elif defined(__WIN32)
// windows implementation ...
#define CLASSIFIER_DATA "some location here..."
#endif

#define VECTOR_FILE "face.vec"
#define SAMPLE_OUTPUT_FILE "face.list"
#define BG_OUTPUT_FILE "bg.list"

// define commonly use type
typedef unsigned int uint32;
typedef unsigned char uint8;

void scanDir(const std::string path, std::vector<std::string>& files,
             const std::vector<std::string> exclusion);

bool fileExists(const std::string filePath);
uint32_t getLineCount(const std::string filePath);
bool createDirectory(const std::string name);
bool deleteFile(const std::string filePath);

#endif /* end of include guard: COMMON_H */
