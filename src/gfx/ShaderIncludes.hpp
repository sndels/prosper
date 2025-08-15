#ifndef PROSPER_GFX_SHADER_INCLUDES_HPP
#define PROSPER_GFX_SHADER_INCLUDES_HPP

#include <filesystem>
#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/hash_set.hpp>
#include <wheels/containers/string.hpp>

namespace gfx
{

// TODO:
// Is this artisinal include parser actually required with all its complexity?
// Wouldn't shaderc::PreprocessGlsl() suffice? If this mess is required,
// document why that is.
void expandIncludes(
    wheels::Allocator &alloc, const std::filesystem::path &currentPath,
    wheels::StrSpan currentSource, wheels::String &fullSource,
    wheels::HashSet<std::filesystem::path> &uniqueIncludes,
    size_t includeDepth);

} // namespace gfx

#endif // PROSPER_GFX_SHADER_INCLUDES_HPP
