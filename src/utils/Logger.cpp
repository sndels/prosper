#include "Logger.hpp"

#include "Utils.hpp"

#include <cstdio>
#include <fmt/printf.h>
#include <wheels/containers/static_array.hpp>

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#else // !_WIN32

// Assume posix
#include <ctime>

#endif // _WIN32

namespace
{

struct Timestamp
{
    int32_t hours{0};
    int32_t minutes{0};
    int32_t seconds{0};
    int32_t millis{0};
};

Timestamp getTimestamp()
{
    Timestamp ret;

    // std::chrono::zoned_time is not yet widely supported and the chrono
    // dance seems to be slow as well. Let's just do this platform specific.
#ifdef _WIN32

    // On Windows, we can get the resolution we want straight from winapi
    SYSTEMTIME currentTime = {};
    GetLocalTime(&currentTime);

    ret = {
        .hours = currentTime.wHour,
        .minutes = currentTime.wMinute,
        .seconds = currentTime.wSecond,
        .millis = currentTime.wMilliseconds,
    };

#else // !_WIN32

    // The standard ctime bits don't have millisecond accuracy so let's go
    // from posix
    timespec currentTime = {};
    if (clock_gettime(CLOCK_REALTIME, &currentTime) != 0)
    {
        fprintf(stderr, "Failed to get clock time\n");
        return ret;
    }

    tm localTime = {};
    if (localtime_r(&currentTime.tv_sec, &localTime) != &localTime)
    {
        fprintf(stderr, "Failed to convert POSIX time to localtime\n");
        return ret;
    }

    ret = {
        .hours = localTime.tm_hour,
        .minutes = localTime.tm_min,
        .seconds = localTime.tm_sec,
        .millis = asserted_cast<int32_t>(currentTime.tv_nsec / 1'000'000),
    };

#endif //!_WIN32

    return ret;
}

const wheels::StaticArray<
    const char *, static_cast<size_t>(utils::detail::LogLevel::Count)>
    sLogLevelNames{{"INFO", "WARN", "ERROR"}};
} // namespace

namespace utils::detail
{

// This wraps fprintf and isolates <windows.h> in the cpp
// NOLINTNEXTLINE(cert-dcl50-cpp)
void vlog(LogLevel level, fmt::string_view fmt, fmt::format_args args)
{
    const Timestamp ts = getTimestamp();
    fmt::print(
        level == LogLevel::Error ? stderr : stdout,
        "{:02d}:{:02d}:{:02d}.{:03d} [{}] {}\n", ts.hours, ts.minutes,
        ts.seconds, ts.millis, sLogLevelNames[(uint8_t)level],
        fmt::vformat(fmt, args));
}

} // namespace utils::detail
