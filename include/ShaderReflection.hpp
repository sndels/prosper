#ifndef PROSPER_SHADER_REFLECTION_HPP
#define PROSPER_SHADER_REFLECTION_HPP

#include <wheels/allocators/allocator.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/span.hpp>

class ShaderReflection
{
  public:
    ShaderReflection(
        wheels::ScopedScratch scopeAlloc, wheels::Allocator &alloc,
        wheels::Span<const uint32_t> spvWords);
    ~ShaderReflection() = default;

    uint32_t pushConstantsBytesize() const;

  private:
    uint32_t _pushConstantsBytesize{0};
};

#endif // PROSPER_SHADER_REFLECTION_HPP
