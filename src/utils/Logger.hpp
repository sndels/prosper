#ifndef PROSPER_UTILS_LOGGER_HPP
#define PROSPER_UTILS_LOGGER_HPP

namespace utils
{

// This wraps fprintf and isolates <windows.h> in the cpp
// NOLINTNEXTLINE(cert-dcl50-cpp)
void zzInternalLogInfo(const char *format...);
// This wraps fprintf and isolates <windows.h> in the cpp
// NOLINTNEXTLINE(cert-dcl50-cpp)
void zzInternalLogWarning(const char *format...);
// This wraps fprintf and isolates <windows.h> in the cpp
// NOLINTNEXTLINE(cert-dcl50-cpp)
void zzInternalLogError(const char *format...);

// Macros in case logger implementation changes at some point
// TODO:
// Use quill or something to get better perf if it becomes an issue?
// First attempt at quill increased compile time quite a bit both on linux and
// windows. On windows, windows.h is pulled into every CU with quill logging,
// costing ~0.5ms front-end time. The architecture of the library doesn't really
// allow it to be quarantined in Logger.cpp. I didn't look much into the linux
// build, but total compile time increased by ~30% in debug builds at least.

} // namespace utils

#define LOG_INFO(...) utils::zzInternalLogInfo(__VA_ARGS__)
#define LOG_WARN(...) utils::zzInternalLogWarning(__VA_ARGS__)
#define LOG_ERR(...) utils::zzInternalLogError(__VA_ARGS__)

#endif // PROSPER_UTILS_LOGGER_HPP
