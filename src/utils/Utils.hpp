#ifndef PROSPER_UTILS_HPP
#define PROSPER_UTILS_HPP

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <wheels/allocators/allocator.hpp>
#include <wheels/allocators/utils.hpp>
#include <wheels/containers/span.hpp>
#include <wheels/containers/string.hpp>

const size_t MAX_FRAMES_IN_FLIGHT = 2;
const size_t MAX_SWAPCHAIN_IMAGES = 8;
const float sIndentPixels = 10.f;
const size_t sMaxMsVertices = 64;
const size_t sMaxMsTriangles = 124;

// Statically casts a into T, asserts that the value fits in T if T is integral
template <typename T, typename V>
constexpr T asserted_cast(V a)
    requires(!std::is_same_v<T, V> && std::is_integral_v<T>)
{
#ifndef NDEBUG
    if (a >= 0)
    {
        if constexpr (std::is_integral_v<V>)
        {
            if constexpr ((sizeof(T) < sizeof(V) ||
                           (sizeof(T) == sizeof(V) && std::is_signed_v<T> &&
                            !std::is_signed_v<V>)))
            {
                if (a > static_cast<V>(std::numeric_limits<T>::max()))
                    WHEELS_ASSERT(!"overflow");
            }
        }
        else
        {
            // NOTE: Some edge cases with large target types will be weird
            // here because of precision and truncation
            if (a > static_cast<V>(std::numeric_limits<T>::max()))
                WHEELS_ASSERT(!"overflow");
            if (a < static_cast<V>(std::numeric_limits<T>::min()))
                WHEELS_ASSERT(!"underflow");
        }
    }
    else if constexpr (!std::numeric_limits<T>::is_signed)
        WHEELS_ASSERT(!"Trying to cast negative into unsigned");
    else
    {
        if constexpr (sizeof(T) < sizeof(V))
        {
            if (a < static_cast<V>(std::numeric_limits<T>::min()))
                WHEELS_ASSERT(!"underflow");
        }
    }
#endif // NDEBUG
    return static_cast<T>(a);
}

std::filesystem::path resPath(const std::filesystem::path &path);
// Returns the path relative from RES_PATH, or the path untouched if it's not
// under it. Considers symlinks to be under the path like any other folder/file.
std::filesystem::path relativePath(const std::filesystem::path &path);
std::filesystem::path binPath(const std::filesystem::path &path);

wheels::String readFileString(
    wheels::Allocator &alloc, const std::filesystem::path &path);

template <typename T> void readRaw(std::ifstream &stream, T &value)
{
    stream.read(reinterpret_cast<char *>(&value), sizeof(value));
}

template <typename T>
void readRawSpan(std::ifstream &stream, wheels::Span<T> span)
{
    stream.read(reinterpret_cast<char *>(span.data()), span.size() * sizeof(T));
}

template <typename T>
void writeRawSpan(std::ofstream &stream, wheels::Span<const T> span)
{
    stream.write(
        reinterpret_cast<const char *>(span.data()), span.size() * sizeof(T));
}

inline void writeRawStrSpan(std::ofstream &stream, const wheels::StrSpan span)
{
    stream.write(span.data(), asserted_cast<std::streamsize>(span.size()));
}

template <typename T> void writeRaw(std::ofstream &stream, const T &value)
{
    stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

inline void appendDefineStr(wheels::String &str, wheels::StrSpan name)
{
    str.extend("#define ");
    str.extend(name);
    str.push_back('\n');
}

inline void appendDefineStr(
    wheels::String &str, wheels::StrSpan name, const char *value)
{
    str.extend("#define ");
    str.extend(name);
    str.push_back(' ');
    str.extend(value);
    str.push_back('\n');
}

template <typename T>
void appendDefineStr(wheels::String &str, wheels::StrSpan name, T value)
{
    str.extend("#define ");
    str.extend(name);
    str.push_back(' ');
    str.extend(std::to_string(value).c_str());
    str.push_back('\n');
}

inline void appendEnumVariantsAsDefines(
    wheels::String &str, wheels::StrSpan prefix,
    wheels::Span<const char *const> names)
{
    uint32_t const size = asserted_cast<uint32_t>(names.size());
    for (uint32_t i = 0u; i < size; ++i)
    {
        str.extend("#define ");
        str.extend(prefix);
        str.push_back('_');
        str.extend(names[i]);
        str.push_back(' ');
        str.extend(std::to_string(i).c_str());
        str.push_back('\n');
    }
}

inline uint32_t getMipCount(uint32_t maxDimension)
{
    WHEELS_ASSERT(maxDimension > 0);
    return 32 - std::countl_zero(maxDimension);
}

// Completes the division by rounding up to the next integer when there is a
// remainder. Assumes both inputs are positive. Returns 0 when dividend is 0.
template <typename T>
    requires std::is_integral_v<T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
T roundedUpQuotient(T dividend, T divisor)
{
    if (dividend == 0) [[unlikely]]
        return 0;

    WHEELS_ASSERT(dividend > 0);
    WHEELS_ASSERT(divisor > 0);

    const T ret = (dividend - 1) / divisor + 1;
    return ret;
}

#define TOKEN_APPEND_HELPER(x, y) x##y
#define TOKEN_APPEND(x, y) TOKEN_APPEND_HELPER(x, y)

// Deferred code execution, runs when the current scope ends
// https://stackoverflow.com/a/42060129
struct DeferDummy
{
};
template <class F> struct Deferrer
{
    F f;
    Deferrer(const F &f)
    : f{f}
    {
    }
    ~Deferrer() { f(); }

    Deferrer(const Deferrer<F> &) = delete;
    Deferrer &operator=(const Deferrer<F> &) = delete;
    Deferrer(Deferrer<F> &&) = delete;
    Deferrer &operator=(Deferrer<F> &&) = delete;
};
template <class F> Deferrer<F> operator*(DeferDummy dummy, F f)
{
    (void)dummy;
    return {f};
}
// Caller supplies the lambda scope, works as intended
// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define defer auto TOKEN_APPEND(zzDefer, __LINE__) = DeferDummy{} *[&]()

void setCurrentThreadName(const char *name);

#endif // PROSPER_UTILS_HPP
