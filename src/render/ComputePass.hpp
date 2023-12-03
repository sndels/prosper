#ifndef PROSPER_RENDER_COMPUTE_PASS_HPP
#define PROSPER_RENDER_COMPUTE_PASS_HPP

#include <functional>
#include <glm/glm.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>
#include <wheels/containers/string.hpp>

#include "../gfx/Device.hpp"
#include "../gfx/Fwd.hpp"
#include "../utils/Fwd.hpp"
#include "../utils/Utils.hpp"

class ComputePass
{
  public:
    struct Shader
    {
        std::filesystem::path relPath;
        wheels::String debugName;
        wheels::Optional<wheels::String> defines;
    };

    ComputePass(
        wheels::ScopedScratch scopeAlloc, Device *device,
        DescriptorAllocator *staticDescriptorsAlloc,
        const std::function<Shader(wheels::Allocator &)>
            &shaderDefinitionCallback,
        uint32_t storageSetIndex = 0,
        wheels::Span<const vk::DescriptorSetLayout> externalDsLayouts = {},
        vk::ShaderStageFlags storageStageFlags =
            vk::ShaderStageFlagBits::eCompute);

    ~ComputePass();

    ComputePass(const ComputePass &other) = delete;
    ComputePass(ComputePass &&other) = delete;
    ComputePass &operator=(const ComputePass &other) = delete;
    ComputePass &operator=(ComputePass &&other) = delete;

    // Returns true if recompile happened
    bool recompileShader(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        const std::function<Shader(wheels::Allocator &)>
            &shaderDefinitionCallback,
        wheels::Span<const vk::DescriptorSetLayout> externalDsLayouts = {});

    template <size_t N>
    void updateDescriptorSet(
        uint32_t nextFrame,
        wheels::StaticArray<DescriptorInfo, N> const &descriptorInfos);

    [[nodiscard]] vk::DescriptorSet storageSet(uint32_t nextFrame) const;
    [[nodiscard]] vk::DescriptorSetLayout storageSetLayout() const;

    void record(
        vk::CommandBuffer cb, const glm::uvec3 &groups,
        wheels::Span<const vk::DescriptorSet> descriptorSets,
        wheels::Span<const uint32_t> dynamicOffsets = {}) const;

    template <typename PCBlock>
    void record(
        vk::CommandBuffer cb, const PCBlock &pcBlock, const glm::uvec3 &groups,
        wheels::Span<const vk::DescriptorSet> descriptorSets,
        wheels::Span<const uint32_t> dynamicOffsets = {}) const;

  private:
    [[nodiscard]] bool compileShader(
        wheels::ScopedScratch scopeAlloc,
        const std::function<Shader(wheels::Allocator &)>
            &shaderDefinitionCallback);

    void destroyPipelines();

    void createDescriptorSets(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc,
        vk::ShaderStageFlags storageStageFlags);

    void createPipeline(
        wheels::ScopedScratch scopeAlloc,
        wheels::Span<const vk::DescriptorSetLayout> externalDsLayouts = {});

    Device *_device{nullptr};

    vk::ShaderModule _shaderModule;
    wheels::Optional<ShaderReflection> _shaderReflection;

    vk::DescriptorSetLayout _storageSetLayout;
    uint32_t _storageSetIndex{0xFFFFFFFF};
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT> _storageSets{
        {}};

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
};

template <size_t N>
void ComputePass::updateDescriptorSet(
    uint32_t nextFrame,
    wheels::StaticArray<DescriptorInfo, N> const &descriptorInfos)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?
    const vk::DescriptorSet ds = _storageSets[nextFrame];

    WHEELS_ASSERT(_shaderReflection.has_value());
    const wheels::StaticArray descriptorWrites =
        _shaderReflection->generateDescriptorWrites(
            _storageSetIndex, ds, descriptorInfos);

    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

template <typename PCBlock>
void ComputePass::record(
    vk::CommandBuffer cb, const PCBlock &pcBlock, const glm::uvec3 &groups,
    wheels::Span<const vk::DescriptorSet> descriptorSets,
    wheels::Span<const uint32_t> dynamicOffsets) const
{
    WHEELS_ASSERT(all(greaterThan(groups, glm::uvec3{0u})));
    WHEELS_ASSERT(_shaderReflection.has_value());
    WHEELS_ASSERT(
        sizeof(PCBlock) == _shaderReflection->pushConstantsBytesize());

    cb.bindPipeline(vk::PipelineBindPoint::eCompute, _pipeline);

    cb.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, // firstSet
        asserted_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
        asserted_cast<uint32_t>(dynamicOffsets.size()), dynamicOffsets.data());

    cb.pushConstants(
        _pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PCBlock),
        &pcBlock);

    cb.dispatch(groups.x, groups.y, groups.z);
}

#endif // PROSPER_RENDER_COMPUTE_PASS_HPP
