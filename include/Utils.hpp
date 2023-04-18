#ifndef PROSPER_CONSTANTS_HPP
#define PROSPER_CONSTANTS_HPP

#include <cassert>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <string>

#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/span.hpp>
#include <wheels/containers/string.hpp>

const size_t MAX_FRAMES_IN_FLIGHT = 2;
const size_t MAX_SWAPCHAIN_IMAGES = 8;

// Statically casts a into T, asserts that the value fits in T if T is integral
template <typename T, typename V> constexpr T asserted_cast(V a)
{
#ifndef NDEBUG
    static_assert(
        !std::is_floating_point<T>::value &&
        "No assertions for floating point target type");

    if constexpr (!std::is_same<T, V>::value && std::is_integral<T>::value)
    {
        if (a >= 0)
        {
            if constexpr (std::is_integral<V>::value)
            {
                if constexpr ((sizeof(T) < sizeof(V) ||
                               (sizeof(T) == sizeof(V) &&
                                std::is_signed<T>::value &&
                                !std::is_signed<V>::value)))
                {
                    if (a > static_cast<V>(std::numeric_limits<T>::max()))
                        assert(!"overflow");
                }
            }
            else
            {
                // NOTE: Some edge cases with large target types will be weird
                // here because of precision and truncation
                if (a > static_cast<V>(std::numeric_limits<T>::max()))
                    assert(!"overflow");
                if (a < static_cast<V>(std::numeric_limits<T>::min()))
                    assert(!"underflow");
            }
        }
        else if constexpr (!std::numeric_limits<T>::is_signed)
            assert(!"Trying to cast negative into unsigned");
        else
        {
            if constexpr (sizeof(T) < sizeof(V))
            {
                if (a < static_cast<V>(std::numeric_limits<T>::min()))
                    assert(!"underflow");
            }
        }
    }
#endif // NDEBUG
    return static_cast<T>(a);
}

std::filesystem::path resPath(const std::filesystem::path &path);
std::filesystem::path binPath(const std::filesystem::path &path);

wheels::String readFileString(
    wheels::Allocator &alloc, const std::filesystem::path &path);

inline void appendDefineStr(wheels::String &str, wheels::StrSpan name)
{
    str.extend("#define ");
    str.extend(name);
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

#endif // PROSPER_CONSTANTS_HPP
