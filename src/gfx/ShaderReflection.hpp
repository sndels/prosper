#ifndef PROSPER_GFX_SHADER_REFLECTION_HPP
#define PROSPER_GFX_SHADER_REFLECTION_HPP

#include <wheels/allocators/allocator.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/hash_map.hpp>
#include <wheels/containers/hash_set.hpp>
#include <wheels/containers/span.hpp>
#include <wheels/containers/string.hpp>

#include <vulkan/vulkan.hpp>

#include <filesystem>
#include <variant>

#include "../utils/Hashes.hpp"
#include "../utils/Utils.hpp"
#include "Fwd.hpp"

struct DescriptorSetMetadata
{
    wheels::String name;
    uint32_t binding{0xFFFFFFFF};
    vk::DescriptorType descriptorType{vk::DescriptorType::eSampler};
    // 0 signals a runtime array
    uint32_t descriptorCount{0xFFFFFFFF};
};

using DescriptorInfo = std::variant<
    vk::DescriptorImageInfo, vk::DescriptorBufferInfo, vk::BufferView,
    wheels::Span<const vk::DescriptorImageInfo>,
    wheels::Span<const vk::DescriptorBufferInfo>>;

class ShaderReflection
{
  public:
    ShaderReflection(wheels::Allocator &alloc) noexcept;
    ~ShaderReflection() = default;

    ShaderReflection(const ShaderReflection &) = delete;
    ShaderReflection(ShaderReflection &&other) noexcept;
    ShaderReflection &operator=(const ShaderReflection &) = delete;
    ShaderReflection &operator=(ShaderReflection &&other) noexcept;

    void init(
        wheels::ScopedScratch scopeAlloc, wheels::Span<const uint32_t> spvWords,
        const wheels::HashSet<std::filesystem::path> &sourcefiles);

    [[nodiscard]] uint32_t pushConstantsBytesize() const;
    [[nodiscard]] wheels::HashMap<
        uint32_t, wheels::Array<DescriptorSetMetadata>> const &
    descriptorSetMetadatas() const;
    [[nodiscard]] const wheels::HashSet<std::filesystem::path> &sourceFiles()
        const;
    [[nodiscard]] bool affected(
        const wheels::HashSet<std::filesystem::path> &changedFiles) const;

    [[nodiscard]] vk::DescriptorSetLayout createDescriptorSetLayout(
        wheels::ScopedScratch scopeAlloc, Device &device,
        uint32_t descriptorSet, vk::ShaderStageFlags stageFlags,
        wheels::Span<const uint32_t> dynamicCounts = {},
        wheels::Span<const vk::DescriptorBindingFlags> bindingFlags = {}) const;

    // TODO: This doesn't deduce N from infos defined as an initializer list
    // Takes bindings sorted by the glsl binding indices. Aliased binds take
    // just one binding.
    [[nodiscard]] wheels::Array<vk::WriteDescriptorSet>
    generateDescriptorWrites(
        wheels::Allocator &alloc, uint32_t descriptorSetIndex,
        vk::DescriptorSet descriptorSetHandle,
        wheels::Span<const DescriptorInfo> descriptorInfos) const;

  private:
    bool _initialized{false};
    wheels::Allocator &_alloc;
    uint32_t _pushConstantsBytesize{0};
    wheels::HashMap<uint32_t, wheels::Array<DescriptorSetMetadata>>
        _descriptorSetMetadatas;
    wheels::HashSet<std::filesystem::path> _sourceFiles;
};

#endif // PROSPER_GFX_SHADER_REFLECTION_HPP
