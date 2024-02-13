#ifndef PROSPER_GFX_SHADER_INCLUDES_HPP
#define PROSPER_GFX_SHADER_INCLUDES_HPP

#include <filesystem>
#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/hash_set.hpp>
#include <wheels/containers/string.hpp>

void expandIncludes(
    wheels::Allocator &alloc, const std::filesystem::path &currentPath,
    wheels::StrSpan currentSource, wheels::String *fullSource,
    wheels::HashSet<std::filesystem::path> *uniqueIncludes,
    size_t includeDepth);

#endif // PROSPER_GFX_SHADER_INCLUDES_HPP
