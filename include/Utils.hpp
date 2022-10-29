#ifndef PROSPER_CONSTANTS_HPP
#define PROSPER_CONSTANTS_HPP

#include <cstddef>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

const size_t MAX_FRAMES_IN_FLIGHT = 2;

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

std::string readFileString(const std::filesystem::path &path);

template <typename T>
constexpr std::string defineStr(const std::string &name, T value)
{
    return "#define " + name + " " + std::to_string(value) + '\n';
}

template <size_t Count>
constexpr std::string enumVariantsAsDefines(
    const std::string &prefix, const std::array<const char *, Count> &names)
{
    std::string ret;
    for (auto i = 0u; i < names.size(); ++i)
        ret += "#define " + prefix + '_' + names[i] + " " + std::to_string(i) +
               '\n';

    return ret;
}

#endif // PROSPER_CONSTANTS_HPP
