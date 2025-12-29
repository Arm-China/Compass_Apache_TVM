// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/include/compass/tvm/runtime/utils.h
 */
#ifndef COMPASS_TVM_RUNTIME_UTILS_H_
#define COMPASS_TVM_RUNTIME_UTILS_H_

#if defined(__QNX__)
#include <limits.h>
#else
#include <linux/limits.h>
#endif

#include <libgen.h>
#include <sys/stat.h>
#include <tvm/runtime/logging.h>
#include <unistd.h>

#include <fstream>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {

inline std::string GetCwd() {
  char buffer[PATH_MAX];
  ICHECK(getcwd(buffer, PATH_MAX) != nullptr);
  return buffer;
}

inline std::string AbsPath(const std::string& path) {
  if (path[0] == '/') return path;
  return (GetCwd() + "/" + path);
}

inline bool IsDir(const std::string& path) {
  struct stat stat_buf;
  if (stat(path.c_str(), &stat_buf) == 0 && S_ISDIR(stat_buf.st_mode)) return true;
  return false;
}

inline bool IsFile(const std::string& path) {
  struct stat stat_buf;
  if (stat(path.c_str(), &stat_buf) == 0 && S_ISREG(stat_buf.st_mode)) return true;
  return false;
}

inline bool IsExecutable(const std::string& path) {
  if (IsFile(path) && access(path.c_str(), X_OK) == 0) return true;
  return false;
}

inline void CreateDirectories(const std::string& path) {
  if (IsDir(path)) return;

  // Create parent directory firstly.
  size_t last_slash_idx = path.find_last_of("/");
  if (last_slash_idx != std::string::npos) {
    CreateDirectories(path.substr(0, last_slash_idx));
  }
  mkdir(path.c_str(), 0777);
  return;
}

inline void ChDir(const std::string& path) {
  ICHECK_EQ(chdir(path.c_str()), 0);
  return;
}

inline std::string Dirname(std::string path) {
  // Here the argument can't be passed as reference, because "dirname" maybe change its content.
  return dirname(const_cast<char*>(path.c_str()));
}

inline void SaveBinary2File(const std::string& file_name, const char* data, size_t num_bytes) {
  // If "num_bytes" equals to 0, this function is just used to create file.
  ICHECK(((num_bytes > 0) && (data != nullptr)) || (num_bytes == 0));

  std::ofstream ofs(file_name, std::ios::binary | std::ios::trunc);
  ICHECK(ofs.fail() == false) << "Can't open or create file \"" << file_name << "\".";
  ofs.write(data, num_bytes);
  return;
}

/*! \brief Return whether a string starts with the given prefix. */
inline bool StrStartsWith(const std::string& str, const std::string& prefix) {
  if (prefix.size() > str.size()) return false;
  return std::equal(str.c_str(), str.c_str() + prefix.size(), prefix.c_str());
}

/*! \brief Return whether a string ends with the given suffix. */
inline bool StrEndsWith(const std::string& str, const std::string& suffix) {
  if (suffix.size() > str.size()) return false;
  return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/*! \brief Return a copy of the string with all occurrences of substring "from"
 *         replaced by "to".
 */
inline std::string StrReplace(const std::string& str, const std::string& from,
                              const std::string& to) {
  std::string ret = str;
  auto pos = ret.find(from);
  while (pos != std::string::npos) {
    ret.replace(pos, from.size(), to);
    pos = ret.find(from, pos + to.size());
  }
  return ret;
}

/*!
 * \brief Return a list of the words in the string, using sep as the delimiter
 *        string.
 */
inline std::vector<std::string> StrSplit(const std::string& str, const std::string& sep) {
  std::string::size_type pos = 0;
  std::string::size_type start = 0;
  std::vector<std::string> ret;
  while ((pos = str.find(sep, start)) != std::string::npos) {
    ret.push_back(str.substr(start, pos - start));
    start = pos + sep.length();
  }
  ret.push_back(str.substr(start));
  return ret;
}

/*! \brief Return a copy of the string converted to lowercase. */
inline std::string StrLower(const std::string& str) {
  std::string ret;
  for (char ch : str) {
    ret.push_back(std::tolower(ch));
  }
  return ret;
}

}  // namespace runtime
}  // namespace tvm
#endif  // COMPASS_TVM_RUNTIME_UTILS_H_
