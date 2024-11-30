#ifndef PROSPER_RENDER_COMPUTE_PASS_HPP
#define PROSPER_RENDER_COMPUTE_PASS_HPP

#include "gfx/Fwd.hpp"
#include "gfx/ShaderReflection.hpp"
#include "utils/Utils.hpp"

#include <functional>
#include <glm/glm.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/inline_array.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/span.hpp>
#include <wheels/containers/static_array.hpp>
#include <wheels/containers/string.hpp>

struct ComputePassOptions
{
    uint32_t storageSetIndex{0};
    uint32_t perFrameRecordLimit{1};
    wheels::Span<const vk::DescriptorSetLayout> externalDsLayouts;
    vk::ShaderStageFlags storageStageFlags{vk::ShaderStageFlagBits::eCompute};
};

struct ComputePassOptionalRecordArgs
{
    wheels::Span<const uint32_t> dynamicOffsets;
    uint32_t specializationIndex{0};
};

class ComputePass
{
  public:
    struct Shader
    {
        std::filesystem::path relPath;
        wheels::String debugName;
        wheels::Optional<wheels::String> defines;
        glm::uvec3 groupSize{16, 16, 1};
    };

    ComputePass() noexcept = default;
    ~ComputePass();

    ComputePass(const ComputePass &other) = delete;
    ComputePass(ComputePass &&other) = delete;
    ComputePass &operator=(const ComputePass &other) = delete;
    ComputePass &operator=(ComputePass &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc,
        const std::function<Shader(wheels::Allocator &)>
            &shaderDefinitionCallback,
        const ComputePassOptions &options = ComputePassOptions{});

    template <typename T>
    void init(
        wheels::ScopedScratch scopeAlloc,
        const std::function<Shader(wheels::Allocator &)>
            &shaderDefinitionCallback,
        wheels::Span<const T> specializationConstants,
        const ComputePassOptions &options = ComputePassOptions{});

    // Returns true if recompile happened
    bool recompileShader(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        const std::function<Shader(wheels::Allocator &)>
            &shaderDefinitionCallback,
        wheels::Span<const vk::DescriptorSetLayout> externalDsLayouts = {});

    // Resets the per frame record count. Doesn't need to be called if
    // perFrameRecordLimit is 1.
    void startFrame();

    // Updates the descriptor for the next record. record() increments the
    // conuter.
    void updateDescriptorSet(
        wheels::ScopedScratch scopeAlloc, uint32_t nextFrame,
        wheels::Span<const DescriptorInfo> descriptorInfos);

    // Returns the descriptor for the next record. record() increments the
    // conuter.
    [[nodiscard]] vk::DescriptorSet storageSet(uint32_t nextFrame) const;
    [[nodiscard]] vk::DescriptorSetLayout storageSetLayout() const;

    // Returns the rounded up group count required to process the input with
    // m_groupSize threads per group
    [[nodiscard]] glm::uvec3 groupCount(glm::uvec3 inputSize) const;

    // Increments the counter for descriptor sets.
    void record(
        vk::CommandBuffer cb, const glm::uvec3 &groupCount,
        wheels::Span<const vk::DescriptorSet> descriptorSets,
        const ComputePassOptionalRecordArgs &optionalArgs = {});

    // Increments the counter for descriptor sets.
    void record(
        vk::CommandBuffer cb, vk::Buffer argumentBuffer,
        wheels::Span<const vk::DescriptorSet> descriptorSets,
        const ComputePassOptionalRecordArgs &optionalArgs = {});

    // Increments the counter for descriptor sets.
    template <typename PCBlock>
    void record(
        vk::CommandBuffer cb, const PCBlock &pcBlock,
        const glm::uvec3 &groupCount,
        wheels::Span<const vk::DescriptorSet> descriptorSets,
        const ComputePassOptionalRecordArgs &optionalArgs = {});

    template <typename PCBlock>
    void record(
        vk::CommandBuffer cb, const PCBlock &pcBlock, vk::Buffer argumentBuffer,
        wheels::Span<const vk::DescriptorSet> descriptorSets,
        const ComputePassOptionalRecordArgs &optionalArgs = {});

  private:
    void init(
        wheels::ScopedScratch scopeAlloc,
        const std::function<Shader(wheels::Allocator &)>
            &shaderDefinitionCallback,
        wheels::Span<const uint8_t> specializationConstants,
        uint32_t specializationConstantsByteSize,
        const ComputePassOptions &options = ComputePassOptions{});

    [[nodiscard]] bool compileShader(
        wheels::ScopedScratch scopeAlloc,
        const Shader &shaderDefinitionCallback);

    void record(
        vk::CommandBuffer cb, wheels::Span<const uint8_t> pcBlockBytes,
        const glm::uvec3 &groupCount,
        wheels::Span<const vk::DescriptorSet> descriptorSets,
        const ComputePassOptionalRecordArgs &optionalArgs = {});

    void record(
        vk::CommandBuffer cb, wheels::Span<const uint8_t> pcBlockBytes,
        vk::Buffer argumentBuffer,
        wheels::Span<const vk::DescriptorSet> descriptorSets,
        const ComputePassOptionalRecordArgs &optionalArgs = {});

    void destroyPipelines();

    void createDescriptorSets(
        wheels::ScopedScratch scopeAlloc, const char *debugName,
        vk::ShaderStageFlags storageStageFlags);

    void createPipelines(
        wheels::ScopedScratch scopeAlloc,
        wheels::Span<const vk::DescriptorSetLayout> externalDsLayouts,
        wheels::StrSpan debugName);

    bool m_initialized{false};

    vk::ShaderModule m_shaderModule;
    wheels::Optional<ShaderReflection> m_shaderReflection;

    vk::DescriptorSetLayout m_storageSetLayout;
    uint32_t m_storageSetIndex{0xFFFF'FFFF};
    // TODO:
    // This much is only needed by FFT. Should use a small vector with less
    // inline space instead?
    static const size_t sPerFrameRecordLimit = 84;
    size_t m_nextRecordIndex{0};
    wheels::StaticArray<
        wheels::InlineArray<vk::DescriptorSet, sPerFrameRecordLimit>,
        MAX_FRAMES_IN_FLIGHT>
        m_storageSets;

    vk::PipelineLayout m_pipelineLayout;
    wheels::Array<uint8_t> m_specializationConstants{gAllocators.general};
    wheels::Array<vk::SpecializationInfo> m_specializationInfos{
        gAllocators.general};
    wheels::Array<vk::Pipeline> m_pipelines{gAllocators.general};

    glm::uvec3 m_groupSize{16, 16, 1};
};

template <typename T>
void ComputePass::init(
    wheels::ScopedScratch scopeAlloc,
    const std::function<Shader(wheels::Allocator &)> &shaderDefinitionCallback,
    wheels::Span<const T> specializationConstants,
    const ComputePassOptions &options)
{
    init(
        WHEELS_MOV(scopeAlloc), shaderDefinitionCallback,
        wheels::Span{
            reinterpret_cast<const uint8_t *>(specializationConstants.data()),
            specializationConstants.size() * sizeof(T)},
        sizeof(T), options);
}

template <typename PCBlock>
void ComputePass::record(
    vk::CommandBuffer cb, const PCBlock &pcBlock, const glm::uvec3 &groupCount,
    wheels::Span<const vk::DescriptorSet> descriptorSets,
    const ComputePassOptionalRecordArgs &optionalArgs)
{
    record(
        cb,
        wheels::Span{
            reinterpret_cast<const uint8_t *>(&pcBlock), sizeof(pcBlock)},
        groupCount, descriptorSets, optionalArgs);
}

template <typename PCBlock>
void ComputePass::record(
    vk::CommandBuffer cb, const PCBlock &pcBlock, vk::Buffer argumentBuffer,
    wheels::Span<const vk::DescriptorSet> descriptorSets,
    const ComputePassOptionalRecordArgs &optionalArgs)
{
    record(
        cb,
        wheels::Span{
            reinterpret_cast<const uint8_t *>(&pcBlock), sizeof(pcBlock)},
        argumentBuffer, descriptorSets, optionalArgs);
}

#endif // PROSPER_RENDER_COMPUTE_PASS_HPP
