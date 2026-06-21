#ifndef PROSPER_UTILS_LOGGER_HPP
#define PROSPER_UTILS_LOGGER_HPP

#include "utils/ForEach.hpp"

#include <cstdint>
#include <fmt/format.h>

namespace utils::detail
{

enum class LogLevel : uint8_t
{
    Info,
    Warning,
    Error,
    Count
};

void vlog(LogLevel level, fmt::string_view fmt, fmt::format_args args);

template <typename... T>
void log(LogLevel level, fmt::format_string<T...> fmt, T &&...args)
{
    vlog(level, fmt.get(), fmt::make_format_args(args...));
}

} // namespace utils::detail

// Macros in case logger implementation changes at some point
// TODO:
// Use quill or something to get better perf if it becomes an issue?
// First attempt at quill increased compile time quite a bit both on linux and
// windows. On windows, windows.h is pulled into every CU with quill logging,
// costing ~0.5ms front-end time. The architecture of the library doesn't really
// allow it to be quarantined in Logger.cpp. I didn't look much into the linux
// build, but total compile time increased by ~30% in debug builds at least.

#define LOG_INFO(fmt, ...)                                                     \
    utils::detail::log(                                                        \
        utils::detail::LogLevel::Info, fmt __VA_OPT__(, ) __VA_ARGS__)
#define LOG_WARN(fmt, ...)                                                     \
    utils::detail::log(                                                        \
        utils::detail::LogLevel::Warning, fmt __VA_OPT__(, ) __VA_ARGS__)
#define LOG_ERR(fmt, ...)                                                      \
    utils::detail::log(                                                        \
        utils::detail::LogLevel::Error, fmt __VA_OPT__(, ) __VA_ARGS__)

#endif // PROSPER_UTILS_LOGGER_HPP
