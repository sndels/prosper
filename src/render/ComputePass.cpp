#include "ComputePass.hpp"

#include "gfx/DescriptorAllocator.hpp"
#include "gfx/Device.hpp"
#include "gfx/VkUtils.hpp"
#include "utils/Logger.hpp"
#include "utils/Utils.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

using namespace glm;
using namespace wheels;

namespace
{

const uint32_t sMaxDynamicOffsets = 8;

}

ComputePass::~ComputePass()
{
    destroyPipelines();
    gDevice.logical().destroy(m_storageSetLayout);
    gDevice.logical().destroy(m_shaderModule);
}

void ComputePass::init(
    wheels::ScopedScratch scopeAlloc,
    const std::function<Shader(wheels::Allocator &)> &shaderDefinitionCallback,
    const ComputePassOptions &options)
{
    init(WHEELS_MOV(scopeAlloc), shaderDefinitionCallback, {}, 0, options);
}

bool ComputePass::recompileShader(
    wheels::ScopedScratch scopeAlloc,
    const wheels::HashSet<std::filesystem::path> &changedFiles,
    const std::function<Shader(wheels::Allocator &)> &shaderDefinitionCallback,
    wheels::Span<const vk::DescriptorSetLayout> externalDsLayouts)
{
    WHEELS_ASSERT(m_initialized);

    WHEELS_ASSERT(m_shaderReflection.has_value());
    if (!m_shaderReflection->affected(changedFiles))
        return false;

    const Shader shader = shaderDefinitionCallback(scopeAlloc);
    if (compileShader(scopeAlloc.child_scope(), shader))
    {
        destroyPipelines();
        createPipelines(
            scopeAlloc.child_scope(), externalDsLayouts, shader.debugName);
        return true;
    }
    return false;
}

void ComputePass::startFrame()
{
    WHEELS_ASSERT(m_initialized);

    m_nextRecordIndex = 0;
}

void ComputePass::updateDescriptorSet(
    ScopedScratch scopeAlloc, uint32_t nextFrame,
    Span<const DescriptorInfo> descriptorInfos)
{
    WHEELS_ASSERT(m_initialized);

    WHEELS_ASSERT(
        m_nextRecordIndex < m_storageSets[nextFrame].size() &&
        "Too many records, forgot to call startFrame() or construct this "
        "ComputePass with enough records?");

    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both groupCount and previous native handle?
    const vk::DescriptorSet ds = m_storageSets[nextFrame][m_nextRecordIndex];

    WHEELS_ASSERT(m_shaderReflection.has_value());
    const wheels::Array descriptorWrites =
        m_shaderReflection->generateDescriptorWrites(
            scopeAlloc, m_storageSetIndex, ds, descriptorInfos);

    gDevice.logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

vk::DescriptorSet ComputePass::storageSet(uint32_t nextFrame) const
{
    WHEELS_ASSERT(m_initialized);

    WHEELS_ASSERT(
        m_nextRecordIndex < m_storageSets[nextFrame].size() &&
        "Too many records, forgot to call startFrame() or construct this "
        "ComputePass with enough records?");

    return m_storageSets[nextFrame][m_nextRecordIndex];
}

vk::DescriptorSetLayout ComputePass::storageSetLayout() const
{
    WHEELS_ASSERT(m_initialized);

    return m_storageSetLayout;
}

uvec3 ComputePass::groupCount(uvec3 inputSize) const
{
    WHEELS_ASSERT(all(greaterThan(inputSize, glm::uvec3{0u})));
    const uvec3 count = (inputSize - 1u) / m_groupSize + 1u;

    return count;
}

void ComputePass::record(
    vk::CommandBuffer cb, const uvec3 &groupCount,
    Span<const vk::DescriptorSet> descriptorSets,
    const ComputePassOptionalRecordArgs &optionalArgs)
{
    WHEELS_ASSERT(m_initialized);

    WHEELS_ASSERT(all(greaterThan(groupCount, glm::uvec3{0u})));
    WHEELS_ASSERT(
        optionalArgs.dynamicOffsets.size() < sMaxDynamicOffsets &&
        "At least some AMD and Intel drivers limit this to 8 per buffer type. "
        "Let's keep the total under if possible to keep things simple.");

    const vk::Pipeline pipeline = m_pipelines[optionalArgs.specializationIndex];

    cb.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);

    cb.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, m_pipelineLayout,
        0, // firstSet
        asserted_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
        asserted_cast<uint32_t>(optionalArgs.dynamicOffsets.size()),
        optionalArgs.dynamicOffsets.data());

    cb.dispatch(groupCount.x, groupCount.y, groupCount.z);

    if (m_storageSets[0].size() > 1)
    {
        // This can equal perFrameRecordLimit if all of them are used
        m_nextRecordIndex++;
    }
}

void ComputePass::record(
    vk::CommandBuffer cb, vk::Buffer argumentBuffer,
    Span<const vk::DescriptorSet> descriptorSets,
    const ComputePassOptionalRecordArgs &optionalArgs)
{
    WHEELS_ASSERT(m_initialized);

    WHEELS_ASSERT(
        optionalArgs.dynamicOffsets.size() < sMaxDynamicOffsets &&
        "At least some AMD and Intel drivers limit this to 8 per buffer type. "
        "Let's keep the total under if possible to keep things simple.");

    const vk::Pipeline pipeline = m_pipelines[optionalArgs.specializationIndex];

    cb.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);

    cb.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, m_pipelineLayout,
        0, // firstSet
        asserted_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
        asserted_cast<uint32_t>(optionalArgs.dynamicOffsets.size()),
        optionalArgs.dynamicOffsets.data());

    cb.dispatchIndirect(argumentBuffer, 0);

    if (m_storageSets[0].size() > 1)
    {
        // This can equal perFrameRecordLimit if all of them are used
        m_nextRecordIndex++;
    }
}

void ComputePass::record(
    vk::CommandBuffer cb, Span<const uint8_t> pcBlockBytes,
    const uvec3 &groupCount, Span<const vk::DescriptorSet> descriptorSets,
    const ComputePassOptionalRecordArgs &optionalArgs)
{
    WHEELS_ASSERT(m_initialized);

    WHEELS_ASSERT(all(greaterThan(groupCount, uvec3{0u})));
    WHEELS_ASSERT(m_shaderReflection.has_value());
    WHEELS_ASSERT(
        pcBlockBytes.size() == m_shaderReflection->pushConstantsBytesize());
    WHEELS_ASSERT(
        optionalArgs.dynamicOffsets.size() < sMaxDynamicOffsets &&
        "At least some AMD and Intel drivers limit this to 8 per buffer type. "
        "Let's keep the total under if possible to keep things simple.");

    const vk::Pipeline pipeline = m_pipelines[optionalArgs.specializationIndex];

    cb.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);

    cb.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, m_pipelineLayout, 0, // firstSet
        asserted_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
        asserted_cast<uint32_t>(optionalArgs.dynamicOffsets.size()),
        optionalArgs.dynamicOffsets.data());

    cb.pushConstants(
        m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
        asserted_cast<uint32_t>(pcBlockBytes.size()), pcBlockBytes.data());

    cb.dispatch(groupCount.x, groupCount.y, groupCount.z);

    if (m_storageSets[0].size() > 1)
    {
        // This can equal perFrameRecordLimit if all of them are used
        m_nextRecordIndex++;
    }
}

void ComputePass::record(
    vk::CommandBuffer cb, wheels::Span<const uint8_t> pcBlockBytes,
    vk::Buffer argumentBuffer,
    wheels::Span<const vk::DescriptorSet> descriptorSets,
    const ComputePassOptionalRecordArgs &optionalArgs)
{
    WHEELS_ASSERT(m_initialized);

    WHEELS_ASSERT(m_shaderReflection.has_value());
    WHEELS_ASSERT(
        pcBlockBytes.size() == m_shaderReflection->pushConstantsBytesize());
    WHEELS_ASSERT(
        optionalArgs.dynamicOffsets.size() < sMaxDynamicOffsets &&
        "At least some AMD and Intel drivers limit this to 8 per buffer type. "
        "Let's keep the total under if possible to keep things simple.");

    const vk::Pipeline pipeline = m_pipelines[optionalArgs.specializationIndex];

    cb.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);

    cb.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, m_pipelineLayout, 0, // firstSet
        asserted_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
        asserted_cast<uint32_t>(optionalArgs.dynamicOffsets.size()),
        optionalArgs.dynamicOffsets.data());

    cb.pushConstants(
        m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
        asserted_cast<uint32_t>(pcBlockBytes.size()), pcBlockBytes.data());

    cb.dispatchIndirect(argumentBuffer, 0);

    if (m_storageSets[0].size() > 1)
    {
        // This can equal perFrameRecordLimit if all of them are used
        m_nextRecordIndex++;
    }
}

void ComputePass::destroyPipelines()
{
    for (const vk::Pipeline pipeline : m_pipelines)
        gDevice.logical().destroy(pipeline);
    gDevice.logical().destroy(m_pipelineLayout);
}

void ComputePass::createDescriptorSets(
    ScopedScratch scopeAlloc, const char *debugName,
    vk::ShaderStageFlags storageStageFlags)
{
    WHEELS_ASSERT(m_shaderReflection.has_value());
    m_storageSetLayout = m_shaderReflection->createDescriptorSetLayout(
        WHEELS_MOV(scopeAlloc), m_storageSetIndex, storageStageFlags);

    for (auto &sets : m_storageSets)
    {
        InlineArray<vk::DescriptorSetLayout, sPerFrameRecordLimit> layouts;
        InlineArray<const char *, sPerFrameRecordLimit> debugNames;
        layouts.resize(sets.size(), m_storageSetLayout);
        debugNames.resize(sets.size(), debugName);
        gStaticDescriptorsAlloc.allocate(layouts, debugNames, sets.mut_span());
    }
}

void ComputePass::createPipelines(
    ScopedScratch scopeAlloc,
    Span<const vk::DescriptorSetLayout> externalDsLayouts, StrSpan debugName)
{
    WHEELS_ASSERT(m_shaderReflection.has_value());

    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = m_shaderReflection->pushConstantsBytesize(),
    };

    WHEELS_ASSERT(m_storageSetIndex == externalDsLayouts.size());
    Array<vk::DescriptorSetLayout> dsLayouts{scopeAlloc};
    dsLayouts.resize(externalDsLayouts.size() + 1);
    if (!externalDsLayouts.empty())
        memcpy(
            dsLayouts.data(), externalDsLayouts.data(),
            externalDsLayouts.size() * sizeof(*externalDsLayouts.data()));
    dsLayouts.back() = m_storageSetLayout;

    m_pipelineLayout =
        gDevice.logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = asserted_cast<uint32_t>(dsLayouts.size()),
            .pSetLayouts = dsLayouts.data(),
            .pushConstantRangeCount = pcRange.size > 0 ? 1u : 0u,
            .pPushConstantRanges = pcRange.size > 0 ? &pcRange : nullptr,

        });

    if (m_specializationInfos.empty())
    {
        const vk::ComputePipelineCreateInfo createInfo{
            .stage =
                {
                    .stage = vk::ShaderStageFlagBits::eCompute,
                    .module = m_shaderModule,
                    .pName = "main",
                },
            .layout = m_pipelineLayout,
        };

        m_pipelines.push_back(createComputePipeline(
            gDevice.logical(), createInfo, debugName.data()));

        return;
    }

    WHEELS_ASSERT(m_specializationInfos.size() < 999);
    const size_t maxCountLen = 3 + 1;
    wheels::String fullDebugName{scopeAlloc};
    fullDebugName.resize(debugName.size() + maxCountLen);
    memcpy(fullDebugName.c_str(), debugName.data(), debugName.size());

    const uint32_t specializationCount = m_specializationInfos.size();
    m_pipelines.reserve(specializationCount);
    for (uint32_t i = 0; i < specializationCount; ++i)
    {
        const vk::SpecializationInfo &info = m_specializationInfos[i];
        const vk::ComputePipelineCreateInfo createInfo{
            .stage =
                {
                    .stage = vk::ShaderStageFlagBits::eCompute,
                    .module = m_shaderModule,
                    .pName = "main",
                    .pSpecializationInfo = &info,
                },
            .layout = m_pipelineLayout,
        };

        snprintf(
            fullDebugName.c_str() + debugName.size(), maxCountLen + 1, "_%d",
            i);
        m_pipelines.push_back(createComputePipeline(
            gDevice.logical(), createInfo, fullDebugName.data()));
    }
}

void ComputePass::init(
    ScopedScratch scopeAlloc,
    const std::function<Shader(wheels::Allocator &)> &shaderDefinitionCallback,
    Span<const uint8_t> specializationConstants,
    uint32_t specializationConstantsByteSize, const ComputePassOptions &options)
{
    WHEELS_ASSERT(
        (options.storageSetIndex == options.externalDsLayouts.size()) &&
        "Implementation assumes that the pass storage set is the last set and "
        "is placed right after the last external one");

    m_storageSetIndex = options.storageSetIndex;

    for (auto &sets : m_storageSets)
        sets.resize(options.perFrameRecordLimit);

    const Shader shader = shaderDefinitionCallback(scopeAlloc);
    LOG_INFO("Creating %s", shader.debugName.c_str());
    if (!compileShader(scopeAlloc.child_scope(), shader))
        throw std::runtime_error("Shader compilation failed");

    if (!specializationConstants.empty())
    {
        WHEELS_ASSERT(
            specializationConstantsByteSize ==
            m_shaderReflection->specializationConstantsByteSize());
        WHEELS_ASSERT(
            specializationConstants.size() % specializationConstantsByteSize ==
            0);

        // Keep track of the constants for shader recompilation
        m_specializationConstants.extend(specializationConstants);

        wheels::Span<const vk::SpecializationMapEntry>
            specializationMapEntries =
                m_shaderReflection->specializationMapEntries();
        const uint32_t specializationMapEntriesCount =
            specializationMapEntries.size();

        m_specializationInfos.reserve(
            m_specializationConstants.size() / specializationConstantsByteSize);
        for (const uint8_t *ptr = m_specializationConstants.data();
             ptr < m_specializationConstants.data() +
                       m_specializationConstants.size();
             ptr += specializationConstantsByteSize)
        {
            m_specializationInfos.push_back(vk::SpecializationInfo{
                .mapEntryCount = specializationMapEntriesCount,
                .pMapEntries = specializationMapEntries.data(),
                .dataSize = specializationConstantsByteSize,
                .pData = ptr,
            });
        }
    }

    createDescriptorSets(
        scopeAlloc.child_scope(), shader.debugName.c_str(),
        options.storageStageFlags);
    createPipelines(
        scopeAlloc.child_scope(), options.externalDsLayouts, shader.debugName);

    m_initialized = true;
}

bool ComputePass::compileShader(
    wheels::ScopedScratch scopeAlloc, const Shader &shader)
{
    WHEELS_ASSERT(all(greaterThan(shader.groupSize, uvec3{0})));
    m_groupSize = shader.groupSize;

    const size_t len =
        56 + (shader.defines.has_value() ? shader.defines->size() : 0);
    String defines{scopeAlloc, len};
    if (shader.defines.has_value())
        defines.extend(*shader.defines);
    appendDefineStr(defines, "GROUP_X", shader.groupSize.x);
    appendDefineStr(defines, "GROUP_Y", shader.groupSize.y);
    appendDefineStr(defines, "GROUP_Z", shader.groupSize.z);
    WHEELS_ASSERT(defines.size() <= len);

    wheels::Optional<Device::ShaderCompileResult> compResult =
        gDevice.compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = shader.relPath,
                                          .debugName = shader.debugName.c_str(),
                                          .defines = defines,
                                      });

    if (compResult.has_value())
    {
        gDevice.logical().destroy(m_shaderModule);

        ShaderReflection &reflection = compResult->reflection;

        m_shaderModule = compResult->module;
        m_shaderReflection = WHEELS_MOV(reflection);

        return true;
    }

    return false;
}
