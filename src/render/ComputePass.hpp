#ifndef PROSPER_RENDER_COMPUTE_PASS_HPP
#define PROSPER_RENDER_COMPUTE_PASS_HPP

#include <glm/glm.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>
#include <wheels/containers/string.hpp>

#include "../gfx/DescriptorAllocator.hpp"
#include "../gfx/Device.hpp"
#include "../utils/Profiler.hpp"
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

    template <typename Callback>
    ComputePass(
        wheels::ScopedScratch scopeAlloc, Device *device,
        DescriptorAllocator *staticDescriptorsAlloc,
        Callback &&shaderDefinitionCallback, uint32_t storageSetIndex = 0,
        wheels::Span<const vk::DescriptorSetLayout> externalDsLayouts = {},
        vk::ShaderStageFlags storageStageFlags =
            vk::ShaderStageFlagBits::eCompute);

    ~ComputePass();

    ComputePass(const ComputePass &other) = delete;
    ComputePass(ComputePass &&other) = delete;
    ComputePass &operator=(const ComputePass &other) = delete;
    ComputePass &operator=(ComputePass &&other) = delete;

    template <typename Callback>
    void recompileShader(
        wheels::ScopedScratch scopeAlloc, Callback &&shaderDefinitionCallback,
        wheels::Span<const vk::DescriptorSetLayout> externalDsLayouts = {});

    template <size_t N>
    void updateDescriptorSet(
        uint32_t nextFrame,
        wheels::StaticArray<DescriptorInfo, N> const &bindingInfos);

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
    template <typename Callback>
    [[nodiscard]] bool compileShader(
        wheels::ScopedScratch scopeAlloc, Callback &&shaderDefinitionCallback);

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

template <typename Callback>
ComputePass::ComputePass(
    wheels::ScopedScratch scopeAlloc, Device *device,
    DescriptorAllocator *staticDescriptorsAlloc,
    Callback &&shaderDefinitionCallback, uint32_t storageSetIndex,
    wheels::Span<const vk::DescriptorSetLayout> externalDsLayouts,
    vk::ShaderStageFlags storageStageFlags)
: _device{device}
, _storageSetIndex{storageSetIndex}
{
    assert(staticDescriptorsAlloc != nullptr);
    assert(
        (_storageSetIndex == externalDsLayouts.size()) &&
        "Implementation assumes that the pass storage set is the last set and "
        "is placed right after the last external one");

    printf("Creating ComputePass\n");
    if (!compileShader(scopeAlloc.child_scope(), shaderDefinitionCallback))
        throw std::runtime_error("Shader compilation failed");

    createDescriptorSets(
        scopeAlloc.child_scope(), staticDescriptorsAlloc, storageStageFlags);
    createPipeline(scopeAlloc.child_scope(), externalDsLayouts);
}

template <typename Callback>
void ComputePass::recompileShader(
    wheels::ScopedScratch scopeAlloc, Callback &&shaderDefinitionCallback,
    wheels::Span<const vk::DescriptorSetLayout> externalDsLayouts)
{
    if (compileShader(scopeAlloc.child_scope(), shaderDefinitionCallback))
    {
        destroyPipelines();
        createPipeline(scopeAlloc.child_scope(), externalDsLayouts);
    }
}

template <size_t N>
void ComputePass::updateDescriptorSet(
    uint32_t nextFrame,
    wheels::StaticArray<DescriptorInfo, N> const &bindingInfos)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?
    const vk::DescriptorSet ds = _storageSets[nextFrame];

    assert(_shaderReflection.has_value());
    const wheels::StaticArray descriptorWrites =
        _shaderReflection->generateDescriptorWrites(
            _storageSetIndex, ds, bindingInfos);

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
    assert(all(greaterThan(groups, glm::uvec3{0u})));
    assert(_shaderReflection.has_value());
    assert(sizeof(PCBlock) == _shaderReflection->pushConstantsBytesize());

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

template <typename Callback>
bool ComputePass::compileShader(
    wheels::ScopedScratch scopeAlloc, Callback &&shaderDefinitionCallback)
{
    Shader shader = shaderDefinitionCallback(scopeAlloc);

    printf("Compiling %s\n", shader.debugName.c_str());

    wheels::Optional<Device::ShaderCompileResult> compResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(),
            Device::CompileShaderModuleArgs{
                .relPath = shader.relPath,
                .debugName = shader.debugName.c_str(),
                .defines = shader.defines.has_value() ? *shader.defines
                                                      : wheels::StrSpan{""},
            });

    if (compResult.has_value())
    {
        _device->logical().destroy(_shaderModule);

        ShaderReflection &reflection = compResult->reflection;

        _shaderModule = compResult->module;
        _shaderReflection = WHEELS_MOV(reflection);

        return true;
    }

    return false;
}

#endif // PROSPER_RENDER_COMPUTE_PASS_HPP
