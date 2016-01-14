#ifndef COMMON_H
#define COMMON_H

#include <opencv2/core.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <cmath>

using std::string;
using std::vector;
using std::ifstream;
using std::ios;

#if defined(__unix__)
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#elif defined(__WIN32)
#include <dirent.h>
#include <windows.h>
#endif

// separator
#if defined(__unix__)
const char* const SEPARATOR = "/";
#elif defined(__WIN32)
// windows implementation ...
const char* const SEPARATOR = "\\";
#endif

// define commonly use type
typedef unsigned int uint32;
typedef unsigned char uint8;

void scanDir(const string path, vector<string>& files,
             const vector<string> exclusion);

bool fileExists(const string filePath);
uint32_t getLineCount(const string filePath);
bool createDirectory(const string name);
bool deleteFile(const string filePath);

#endif /* end of include guard: COMMON_H */
