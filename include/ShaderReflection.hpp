#ifndef PROSPER_SHADER_REFLECTION_HPP
#define PROSPER_SHADER_REFLECTION_HPP

#include <wheels/allocators/allocator.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/hash_map.hpp>
#include <wheels/containers/span.hpp>
#include <wheels/containers/static_array.hpp>
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
    ShaderReflection(wheels::Allocator &alloc);
    ShaderReflection(
        wheels::ScopedScratch scopeAlloc, wheels::Allocator &alloc,
        wheels::Span<const uint32_t> spvWords);
    ~ShaderReflection() = default;

    ShaderReflection(const ShaderReflection &) = delete;
    ShaderReflection(ShaderReflection &&) = default;
    ShaderReflection &operator=(const ShaderReflection &) = delete;
    ShaderReflection &operator=(ShaderReflection &&) = default;

    [[nodiscard]] uint32_t pushConstantsBytesize() const;
    [[nodiscard]] wheels::HashMap<
        uint32_t, wheels::Array<DescriptorSetMetadata>> const &
    descriptorSetMetadatas() const;

    [[nodiscard]] wheels::Array<vk::DescriptorSetLayoutBinding>
    generateLayoutBindings(
        wheels::Allocator &alloc, uint32_t descriptorSet,
        vk::ShaderStageFlags stageFlags) const;

    // TODO: This doesn't deduce N from infos defined as an initializer list
    template <size_t N>
    [[nodiscard]] wheels::StaticArray<vk::WriteDescriptorSet, N>
    generateDescriptorWrites(
        uint32_t descriptorSetIndex, vk::DescriptorSet descriptorSetHandle,
        wheels::StaticArray<
            wheels::Pair<uint32_t, const vk::DescriptorImageInfo *>, N>
            bindingImageInfos) const;

  private:
    uint32_t _pushConstantsBytesize{0};
    wheels::HashMap<uint32_t, wheels::Array<DescriptorSetMetadata>>
        _descriptorSetMetadatas;
};

template <size_t N>
wheels::StaticArray<vk::WriteDescriptorSet, N> ShaderReflection::
    generateDescriptorWrites(
        uint32_t descriptorSetIndex, vk::DescriptorSet descriptorSetHandle,
        wheels::StaticArray<
            wheels::Pair<uint32_t, const vk::DescriptorImageInfo *>, N>
            bindingImageInfos) const
{
    const wheels::Array<DescriptorSetMetadata> *metadatas =
        _descriptorSetMetadatas.find(descriptorSetIndex);
    assert(metadatas != nullptr);

    wheels::StaticArray<vk::WriteDescriptorSet, N> descriptorWrites;
    for (const auto &bindingInfo : bindingImageInfos)
    {
        uint32_t binding = bindingInfo.first;
        const vk::DescriptorImageInfo *pImageInfo = bindingInfo.second;

        bool found = false;
        for (const DescriptorSetMetadata &metadata : *metadatas)
        {
            if (metadata.binding == binding)
            {
                found = true;
                descriptorWrites.push_back(vk::WriteDescriptorSet{
                    .dstSet = descriptorSetHandle,
                    .dstBinding = binding,
                    .descriptorCount = 1,
                    .descriptorType = metadata.descriptorType,
                    .pImageInfo = pImageInfo,
                });
            }
        }
        assert(found && "Binding index not found");
    }

    return descriptorWrites;
}

#endif // PROSPER_SHADER_REFLECTION_HPP
