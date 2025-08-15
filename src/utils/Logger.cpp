#include "Logger.hpp"

#include "Utils.hpp"

#include <cstdarg>
#include <cstdio>
#include <wheels/allocators/utils.hpp>
#include <wheels/assert.hpp>
#include <wheels/containers/static_array.hpp>

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#else // !_WIN32

// Assume posix
#include <ctime>

#endif // _WIN32

using namespace wheels;

// va_list is not specified by the standard
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-array-to-pointer-decay)

namespace utils
{

namespace
{

// This should be plenty for even the worst shader error messes and still small
// enough to comfortably fit the 1MB stack on windows.
const int sTmpStrLength = static_cast<int>(kilobytes(8));
const StaticArray sOutOfSpaceError =
    "\n[ ERROR: Logger ran out of tmp formatting space ]\n";

// TODO:
// InlineArray and resize_uninitialized() instead?
struct TmpStr
{
    StaticArray<char, sTmpStrLength> str;
    size_t writePtr{0};
    bool outOfSpace{false};

    void appendTimestamp()
    {
        // std::chrono::zoned_time is not yet widely supported and the chrono
        // dance seems to be slow as well. Let's just do this platform specific.
#ifdef _WIN32

        // On Windows, we can get the resolution we want straight from winapi
        SYSTEMTIME currentTime = {};
        GetLocalTime(&currentTime);

        const int32_t hours = currentTime.wHour;
        const int32_t minutes = currentTime.wMinute;
        const int32_t seconds = currentTime.wSecond;
        const int32_t millis = currentTime.wMilliseconds;

#else // !_WIN32

        // The standard ctime bits don't have millisecond accuracy so let's go
        // from posix
        timespec currentTime = {};
        if (clock_gettime(CLOCK_REALTIME, &currentTime) != 0)
        {
            fprintf(stderr, "Failed to get clock time");
            appendWithoutNewline("HH:MM:SS.mmm ");
            return;
        }

        tm localTime = {};
        if (localtime_r(&currentTime.tv_sec, &localTime) != &localTime)
        {
            fprintf(stderr, "Failed to convert POSIX time to localtime");
            appendWithoutNewline("HH:MM:SS.mmm ");
            return;
        }

        const int32_t hours = localTime.tm_hour;
        const int32_t minutes = localTime.tm_min;
        const int32_t seconds = localTime.tm_sec;
        const int32_t millis =
            asserted_cast<int32_t>(currentTime.tv_nsec / 1'000'000);

#endif //!_WIN32

        appendWithoutNewline(
            "%02d:%02d:%02d.%03d ", hours, minutes, seconds, millis);
    }

    void appendImpl(bool appendNewline, const char *format, va_list args)
    {
        if (outOfSpace)
            return;

        // Include final null after the overflow error
        const int space = sTmpStrLength - asserted_cast<int>(writePtr) -
                          asserted_cast<int>(sOutOfSpaceError.size()) - 1;
        WHEELS_ASSERT(space >= 0);

        const int charsWritten =
            vsnprintf(str.data() + writePtr, space, format, args);
        WHEELS_ASSERT(charsWritten > 0 && "Encoding error");

        if (charsWritten < space)
            writePtr += charsWritten;
        else
        {
            // Let's just print the whole size even if it's garbage, but replace
            // null(s) with spaces so that we'll still print the overflow error
            // afterward
            for (char &c : str)
            {
                if (c == '\0')
                    c = ' ';
            }
            writePtr += space;
            outOfSpace = true;
        }

        if (appendNewline && !outOfSpace)
        {
            str[writePtr++] = '\n';
            str[writePtr++] = '\0';
        }
    }

    void appendWithNewline(const char *format, va_list args)
    {
        appendImpl(true, format, args);
    }

    // This wraps fprintf and isolates <windows.h> in the cpp
    // NOLINTNEXTLINE(cert-dcl50-cpp)
    void appendWithNewline(const char *format...)
    {
        va_list args = {};
        va_start(args, format);

        appendImpl(true, format, args);

        va_end(args);
    }

    // This wraps fprintf and isolates <windows.h> in the cpp
    // NOLINTNEXTLINE(cert-dcl50-cpp)
    void appendWithoutNewline(const char *format...)
    {
        va_list args = {};
        va_start(args, format);

        appendImpl(false, format, args);

        va_end(args);
    }

    void checkAndHandleOverflow()
    {
        if (outOfSpace)
        {
            WHEELS_ASSERT(
                sTmpStrLength - asserted_cast<int>(writePtr) >=
                asserted_cast<int>(sOutOfSpaceError.size()) + 1);
            // Copy the final null too
            memcpy(
                str.data() + writePtr, sOutOfSpaceError.data(),
                asserted_cast<int>(sOutOfSpaceError.size()) + 1);
        }
    }
};

} // namespace

// This wraps fprintf and isolates <windows.h> in the cpp
// NOLINTNEXTLINE(cert-dcl50-cpp)
void zzInternalLogInfo(const char *format...)
{
    TmpStr tmpStr;
    tmpStr.appendTimestamp();
    tmpStr.appendWithoutNewline("[INFO]  ");

    va_list args = {};
    va_start(args, format);
    tmpStr.appendWithNewline(format, args);
    va_end(args);

    tmpStr.checkAndHandleOverflow();

    // PERFNOTE:
    // fprintf dominates this on windows at least.

    fprintf(stdout, "%s", tmpStr.str.data());
#ifdef _WIN32
    // Also output to debug output for convenience
    OutputDebugStringA(tmpStr.str.data());
#endif //_WIN32
}

// This wraps fprintf and isolates <windows.h> in the cpp
// NOLINTNEXTLINE(cert-dcl50-cpp)
void zzInternalLogWarning(const char *format...)
{
    TmpStr tmpStr;
    tmpStr.appendTimestamp();
    tmpStr.appendWithoutNewline("[WARN]  ");

    va_list args = {};
    va_start(args, format);
    tmpStr.appendWithNewline(format, args);
    va_end(args);

    tmpStr.checkAndHandleOverflow();

    fprintf(stdout, "%s", tmpStr.str.data());
#ifdef _WIN32
    // Also output to debug output for convenience
    OutputDebugStringA(tmpStr.str.data());
#endif //_WIN32
}

// This wraps fprintf and isolates <windows.h> in the cpp
// NOLINTNEXTLINE(cert-dcl50-cpp)
void zzInternalLogError(const char *format...)
{
    TmpStr tmpStr;
    tmpStr.appendTimestamp();
    tmpStr.appendWithoutNewline("[ERROR] ");

    va_list args = {};
    va_start(args, format);
    tmpStr.appendWithNewline(format, args);
    va_end(args);

    tmpStr.checkAndHandleOverflow();

    fprintf(stderr, "%s", tmpStr.str.data());
#ifdef _WIN32
    // Also output to debug output for convenience
    OutputDebugStringA(tmpStr.str.data());
#endif //_WIN32
}

// NOLINTEND(cppcoreguidelines-pro-bounds-array-to-pointer-decay)

} // namespace utils
