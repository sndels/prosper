#ifndef PROSPER_GFX_SHADER_REFLECTION_HPP
#define PROSPER_GFX_SHADER_REFLECTION_HPP

#include <wheels/allocators/allocator.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/hash_map.hpp>
#include <wheels/containers/span.hpp>
#include <wheels/containers/static_array.hpp>
#include <wheels/containers/string.hpp>

#include <vulkan/vulkan.hpp>

#include <variant>

#include "../utils/Utils.hpp"

class Device;

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

    [[nodiscard]] vk::DescriptorSetLayout createDescriptorSetLayout(
        wheels::ScopedScratch scopeAlloc, Device &device,
        uint32_t descriptorSet, vk::ShaderStageFlags stageFlags,
        wheels::Span<const uint32_t> dynamicCounts = {},
        wheels::Span<const vk::DescriptorBindingFlags> bindingFlags = {}) const;

    // TODO: This doesn't deduce N from infos defined as an initializer list
    // Takes bindings sorted by the glsl binding indices
    template <size_t N>
    [[nodiscard]] wheels::StaticArray<vk::WriteDescriptorSet, N>
    generateDescriptorWrites(
        uint32_t descriptorSetIndex, vk::DescriptorSet descriptorSetHandle,
        const wheels::StaticArray<DescriptorInfo, N> &bindingInfos) const;

  private:
    uint32_t _pushConstantsBytesize{0};
    wheels::HashMap<uint32_t, wheels::Array<DescriptorSetMetadata>>
        _descriptorSetMetadatas;
};

template <size_t N>
wheels::StaticArray<vk::WriteDescriptorSet, N> ShaderReflection::
    generateDescriptorWrites(
        uint32_t descriptorSetIndex, vk::DescriptorSet descriptorSetHandle,
        const wheels::StaticArray<DescriptorInfo, N> &bindingInfos) const
{
    const wheels::Array<DescriptorSetMetadata> *metadatas =
        _descriptorSetMetadatas.find(descriptorSetIndex);
    assert(metadatas != nullptr);
    assert(metadatas->size() == N);

    wheels::StaticArray<vk::WriteDescriptorSet, N> descriptorWrites;
    assert(bindingInfos.size() == N);

    for (uint32_t i = 0; i < N; ++i)
    {
        const DescriptorInfo &descriptorInfo = bindingInfos[i];

        const vk::DescriptorImageInfo *pImageInfo =
            std::get_if<vk::DescriptorImageInfo>(&descriptorInfo);
        const vk::DescriptorBufferInfo *pBufferInfo =
            std::get_if<vk::DescriptorBufferInfo>(&descriptorInfo);
        const vk::BufferView *pTexelBufferView =
            std::get_if<vk::BufferView>(&descriptorInfo);
        // TODO:
        // Refactor this so that single image is also a span? How are the
        // ergonomics?
        const wheels::Span<const vk::DescriptorImageInfo> *pImageInfoSpan =
            std::get_if<wheels::Span<const vk::DescriptorImageInfo>>(
                &descriptorInfo);
        const wheels::Span<const vk::DescriptorBufferInfo> *pBufferInfoSpan =
            std::get_if<wheels::Span<const vk::DescriptorBufferInfo>>(
                &descriptorInfo);

        uint32_t descriptorCount = 1;

        if (pImageInfoSpan != nullptr)
        {
            pImageInfo = pImageInfoSpan->data();
            descriptorCount = asserted_cast<uint32_t>(pImageInfoSpan->size());
        }
        else if (pBufferInfoSpan != nullptr)
        {
            pBufferInfo = pBufferInfoSpan->data();
            descriptorCount = asserted_cast<uint32_t>(pBufferInfoSpan->size());
        }

        const DescriptorSetMetadata &metadata = (*metadatas)[i];
        descriptorWrites.push_back(vk::WriteDescriptorSet{
            .dstSet = descriptorSetHandle,
            .dstBinding = metadata.binding,
            .descriptorCount = descriptorCount,
            .descriptorType = metadata.descriptorType,
            .pImageInfo = pImageInfo,
            .pBufferInfo = pBufferInfo,
            .pTexelBufferView = pTexelBufferView,
        });
    }

    return descriptorWrites;
}

#endif // PROSPER_GFX_SHADER_REFLECTION_HPP
