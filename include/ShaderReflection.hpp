#ifndef PROSPER_SHADER_REFLECTION_HPP
#define PROSPER_SHADER_REFLECTION_HPP

#include <wheels/allocators/allocator.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/hash_map.hpp>
#include <wheels/containers/span.hpp>
#include <wheels/containers/string.hpp>

#include <vulkan/vulkan.hpp>

struct DescriptorSetMetadata
{
    wheels::String name;
    uint32_t binding{0xFFFFFFFF};
    vk::DescriptorType descriptorType{vk::DescriptorType::eSampler};
    // 0 signals a runtime array
    uint32_t descriptorCount{0xFFFFFFFF};
};

class ShaderReflection
{
  public:
    ShaderReflection(
        wheels::ScopedScratch scopeAlloc, wheels::Allocator &alloc,
        wheels::Span<const uint32_t> spvWords);
    ~ShaderReflection() = default;

    ShaderReflection(const ShaderReflection &) = delete;
    ShaderReflection(ShaderReflection &&) = default;
    ShaderReflection &operator=(const ShaderReflection &) = delete;
    ShaderReflection &operator=(ShaderReflection &&) = default;

    uint32_t pushConstantsBytesize() const;
    wheels::HashMap<uint32_t, wheels::Array<DescriptorSetMetadata>> const &
    descriptorSetMetadatas() const;

  private:
    uint32_t _pushConstantsBytesize{0};
    wheels::HashMap<uint32_t, wheels::Array<DescriptorSetMetadata>>
        _descriptorSetMetadatas;
};

#endif // PROSPER_SHADER_REFLECTION_HPP
