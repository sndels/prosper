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
    wheels::Span<const vk::DescriptorImageInfo>>;

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
        vk::ShaderStageFlags stageFlags,
        wheels::Span<const uint32_t> dynamicCounts = {}) const;

    // TODO: This doesn't deduce N from infos defined as an initializer list
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

    wheels::StaticArray<vk::WriteDescriptorSet, N> descriptorWrites;
    assert(bindingInfos.size() == N);
    for (uint32_t binding = 0; binding < N; ++binding)
    {
        const DescriptorInfo &descriptorInfo = bindingInfos[binding];

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

        uint32_t descriptorCount = 1;

        if (pImageInfoSpan != nullptr)
        {
            pImageInfo = pImageInfoSpan->data();
            descriptorCount = asserted_cast<uint32_t>(pImageInfoSpan->size());
        }

        bool found = false;
        for (const DescriptorSetMetadata &metadata : *metadatas)
        {
            if (metadata.binding == binding)
            {
                found = true;
                descriptorWrites.push_back(vk::WriteDescriptorSet{
                    .dstSet = descriptorSetHandle,
                    .dstBinding = binding,
                    .descriptorCount = descriptorCount,
                    .descriptorType = metadata.descriptorType,
                    .pImageInfo = pImageInfo,
                    .pBufferInfo = pBufferInfo,
                    .pTexelBufferView = pTexelBufferView,
                });
            }
        }
        assert(found && "Binding index not found");
        // Supress unused variable warning. Let's not mask found with NDEBUG
        // macro spaghetti since the logic is not expensive.
        (void)found;
    }

    return descriptorWrites;
}

#endif // PROSPER_GFX_SHADER_REFLECTION_HPP
