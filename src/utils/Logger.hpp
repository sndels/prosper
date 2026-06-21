#ifndef PROSPER_UTILS_LOGGER_HPP
#define PROSPER_UTILS_LOGGER_HPP

#include <cstdint>

namespace utils::detail
{

enum class LogLevel : uint8_t
{
    Info,
    Warning,
    Error
};

// This wraps fprintf and isolates <windows.h> in the cpp
// NOLINTNEXTLINE(cert-dcl50-cpp)
void log(LogLevel level, const char *format...);

} // namespace utils::detail

// Macros in case logger implementation changes at some point
// TODO:
// Use quill or something to get better perf if it becomes an issue?
// First attempt at quill increased compile time quite a bit both on linux and
// windows. On windows, windows.h is pulled into every CU with quill logging,
// costing ~0.5ms front-end time. The architecture of the library doesn't really
// allow it to be quarantined in Logger.cpp. I didn't look much into the linux
// build, but total compile time increased by ~30% in debug builds at least.

#define LOG_INFO(...)                                                          \
    utils::detail::log(utils::detail::LogLevel::Info, __VA_ARGS__)
#define LOG_WARN(...)                                                          \
    utils::detail::log(utils::detail::LogLevel::Warning, __VA_ARGS__)
#define LOG_ERR(...)                                                           \
    utils::detail::log(utils::detail::LogLevel::Error, __VA_ARGS__)

#endif // PROSPER_UTILS_LOGGER_HPP
