#ifndef PROSPER_UTILS_HASHES_HPP
#define PROSPER_UTILS_HASHES_HPP

// include-cleaner can't figure out when HashMap/HashSet require these
// implementations
// IWYU pragma: always_keep

#include <filesystem>
#include <wheels/containers/hash.hpp>
#include <wheels/containers/span.hpp>

namespace wheels
{

// TODO:
// Implement in wheels? Should ensure that this matches between a string and its
// full span.
template <> struct Hash<StrSpan>
{
    [[nodiscard]] uint64_t operator()(StrSpan const &value) const noexcept
    {
        return wyhash(value.data(), value.size(), 0, (uint64_t const *)_wyp);
    }
};

template <> struct Hash<std::filesystem::path>
{
    [[nodiscard]] uint64_t operator()(
        const std::filesystem::path &value) const noexcept
    {
        return wyhash(
            value.string().c_str(), value.string().size(), 0,
            (uint64_t const *)_wyp);
    }
};

} // namespace wheels

#endif // PROSPER_UTILS_HASHES_HPP
