#include "ComputePass.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include "../gfx/DescriptorAllocator.hpp"
#include "../gfx/Device.hpp"
#include "../gfx/VkUtils.hpp"
#include "../utils/Utils.hpp"
#include "RenderTargets.hpp"

using namespace glm;
using namespace wheels;

namespace
{

const uint32_t sMaxDynamicOffsets = 8;

}

ComputePass::~ComputePass()
{
    destroyPipelines();
    gDevice.logical().destroy(_storageSetLayout);
    gDevice.logical().destroy(_shaderModule);
}

void ComputePass::init(
    wheels::ScopedScratch scopeAlloc,
    DescriptorAllocator *staticDescriptorsAlloc,
    const std::function<Shader(wheels::Allocator &)> &shaderDefinitionCallback,
    const ComputePassOptions &options)
{
    WHEELS_ASSERT(!_initialized);
    WHEELS_ASSERT(staticDescriptorsAlloc != nullptr);
    WHEELS_ASSERT(
        (options.storageSetIndex == options.externalDsLayouts.size()) &&
        "Implementation assumes that the pass storage set is the last set and "
        "is placed right after the last external one");

    _storageSetIndex = options.storageSetIndex;

    for (auto &sets : _storageSets)
        sets.resize(options.perFrameRecordLimit);

    const Shader shader = shaderDefinitionCallback(scopeAlloc);
    printf("Creating %s\n", shader.debugName.c_str());
    if (!compileShader(scopeAlloc.child_scope(), shader))
        throw std::runtime_error("Shader compilation failed");

    createDescriptorSets(
        scopeAlloc.child_scope(), staticDescriptorsAlloc,
        shader.debugName.c_str(), options.storageStageFlags);
    createPipeline(
        scopeAlloc.child_scope(), options.externalDsLayouts, shader.debugName);

    _initialized = true;
}

bool ComputePass::recompileShader(
    wheels::ScopedScratch scopeAlloc,
    const wheels::HashSet<std::filesystem::path> &changedFiles,
    const std::function<Shader(wheels::Allocator &)> &shaderDefinitionCallback,
    wheels::Span<const vk::DescriptorSetLayout> externalDsLayouts)
{
    WHEELS_ASSERT(_initialized);

    WHEELS_ASSERT(_shaderReflection.has_value());
    if (!_shaderReflection->affected(changedFiles))
        return false;

    const Shader shader = shaderDefinitionCallback(scopeAlloc);
    if (compileShader(scopeAlloc.child_scope(), shader))
    {
        destroyPipelines();
        createPipeline(
            scopeAlloc.child_scope(), externalDsLayouts, shader.debugName);
        return true;
    }
    return false;
}

void ComputePass::startFrame()
{
    WHEELS_ASSERT(_initialized);

    _nextRecordIndex = 0;
}

void ComputePass::updateDescriptorSet(
    ScopedScratch scopeAlloc, uint32_t nextFrame,
    Span<const DescriptorInfo> descriptorInfos)
{
    WHEELS_ASSERT(_initialized);

    WHEELS_ASSERT(
        _nextRecordIndex < _storageSets[nextFrame].size() &&
        "Too many records, forgot to call startFrame() or construct this "
        "ComputePass with enough records?");

    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?
    const vk::DescriptorSet ds = _storageSets[nextFrame][_nextRecordIndex];

    WHEELS_ASSERT(_shaderReflection.has_value());
    const wheels::Array descriptorWrites =
        _shaderReflection->generateDescriptorWrites(
            scopeAlloc, _storageSetIndex, ds, descriptorInfos);

    gDevice.logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

vk::DescriptorSet ComputePass::storageSet(uint32_t nextFrame) const
{
    WHEELS_ASSERT(_initialized);

    WHEELS_ASSERT(
        _nextRecordIndex < _storageSets[nextFrame].size() &&
        "Too many records, forgot to call startFrame() or construct this "
        "ComputePass with enough records?");

    return _storageSets[nextFrame][_nextRecordIndex];
}

vk::DescriptorSetLayout ComputePass::storageSetLayout() const
{
    WHEELS_ASSERT(_initialized);

    return _storageSetLayout;
}

void ComputePass::record(
    vk::CommandBuffer cb, const uvec3 &extent,
    Span<const vk::DescriptorSet> descriptorSets,
    wheels::Span<const uint32_t> dynamicOffsets)
{
    WHEELS_ASSERT(_initialized);

    WHEELS_ASSERT(all(greaterThan(extent, glm::uvec3{0u})));
    WHEELS_ASSERT(
        dynamicOffsets.size() < sMaxDynamicOffsets &&
        "At least some AMD and Intel drivers limit this to 8 per buffer type. "
        "Let's keep the total under if possible to keep things simple.");

    cb.bindPipeline(vk::PipelineBindPoint::eCompute, _pipeline);

    cb.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, _pipelineLayout,
        0, // firstSet
        asserted_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
        asserted_cast<uint32_t>(dynamicOffsets.size()), dynamicOffsets.data());

    const uvec3 groups = (extent - 1u) / _groupSize + 1u;

    cb.dispatch(groups.x, groups.y, groups.z);

    if (_storageSets[0].size() > 1)
    {
        // This can equal perFrameRecordLimit if all of them are used
        _nextRecordIndex++;
    }
}

void ComputePass::record(
    vk::CommandBuffer cb, vk::Buffer argumentBuffer,
    Span<const vk::DescriptorSet> descriptorSets,
    wheels::Span<const uint32_t> dynamicOffsets)
{
    WHEELS_ASSERT(_initialized);

    WHEELS_ASSERT(
        dynamicOffsets.size() < sMaxDynamicOffsets &&
        "At least some AMD and Intel drivers limit this to 8 per buffer type. "
        "Let's keep the total under if possible to keep things simple.");

    cb.bindPipeline(vk::PipelineBindPoint::eCompute, _pipeline);

    cb.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, _pipelineLayout,
        0, // firstSet
        asserted_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
        asserted_cast<uint32_t>(dynamicOffsets.size()), dynamicOffsets.data());

    cb.dispatchIndirect(argumentBuffer, 0);

    if (_storageSets[0].size() > 1)
    {
        // This can equal perFrameRecordLimit if all of them are used
        _nextRecordIndex++;
    }
}

void ComputePass::record(
    vk::CommandBuffer cb, Span<const uint8_t> pcBlockBytes, const uvec3 &extent,
    Span<const vk::DescriptorSet> descriptorSets,
    Span<const uint32_t> dynamicOffsets)
{
    WHEELS_ASSERT(_initialized);

    WHEELS_ASSERT(all(greaterThan(extent, uvec3{0u})));
    WHEELS_ASSERT(_shaderReflection.has_value());
    WHEELS_ASSERT(
        pcBlockBytes.size() == _shaderReflection->pushConstantsBytesize());
    WHEELS_ASSERT(
        dynamicOffsets.size() < sMaxDynamicOffsets &&
        "At least some AMD and Intel drivers limit this to 8 per buffer type. "
        "Let's keep the total under if possible to keep things simple.");

    cb.bindPipeline(vk::PipelineBindPoint::eCompute, _pipeline);

    cb.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, // firstSet
        asserted_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
        asserted_cast<uint32_t>(dynamicOffsets.size()), dynamicOffsets.data());

    cb.pushConstants(
        _pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
        asserted_cast<uint32_t>(pcBlockBytes.size()), pcBlockBytes.data());

    const uvec3 groups = (extent - 1u) / _groupSize + 1u;

    cb.dispatch(groups.x, groups.y, groups.z);

    if (_storageSets[0].size() > 1)
    {
        // This can equal perFrameRecordLimit if all of them are used
        _nextRecordIndex++;
    }
}

void ComputePass::destroyPipelines()
{
    gDevice.logical().destroy(_pipeline);
    gDevice.logical().destroy(_pipelineLayout);
}

void ComputePass::createDescriptorSets(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc,
    const char *debugName, vk::ShaderStageFlags storageStageFlags)
{
    WHEELS_ASSERT(_shaderReflection.has_value());
    _storageSetLayout = _shaderReflection->createDescriptorSetLayout(
        WHEELS_MOV(scopeAlloc), _storageSetIndex, storageStageFlags);

    for (auto &sets : _storageSets)
    {
        InlineArray<vk::DescriptorSetLayout, sPerFrameRecordLimit> layouts;
        InlineArray<const char *, sPerFrameRecordLimit> debugNames;
        layouts.resize(sets.size(), _storageSetLayout);
        debugNames.resize(sets.size(), debugName);
        staticDescriptorsAlloc->allocate(layouts, debugNames, sets.mut_span());
    }
}

void ComputePass::createPipeline(
    wheels::ScopedScratch scopeAlloc,
    Span<const vk::DescriptorSetLayout> externalDsLayouts, StrSpan debugName)
{
    WHEELS_ASSERT(_shaderReflection.has_value());

    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = _shaderReflection->pushConstantsBytesize(),
    };

    WHEELS_ASSERT(_storageSetIndex == externalDsLayouts.size());
    Array<vk::DescriptorSetLayout> dsLayouts{scopeAlloc};
    dsLayouts.resize(externalDsLayouts.size() + 1);
    if (!externalDsLayouts.empty())
        memcpy(
            dsLayouts.data(), externalDsLayouts.data(),
            externalDsLayouts.size() * sizeof(*externalDsLayouts.data()));
    dsLayouts.back() = _storageSetLayout;

    _pipelineLayout =
        gDevice.logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = asserted_cast<uint32_t>(dsLayouts.size()),
            .pSetLayouts = dsLayouts.data(),
            .pushConstantRangeCount = pcRange.size > 0 ? 1u : 0u,
            .pPushConstantRanges = pcRange.size > 0 ? &pcRange : nullptr,
        });

    const vk::ComputePipelineCreateInfo createInfo{
        .stage =
            {
                .stage = vk::ShaderStageFlagBits::eCompute,
                .module = _shaderModule,
                .pName = "main",
            },
        .layout = _pipelineLayout,
    };

    _pipeline =
        createComputePipeline(gDevice.logical(), createInfo, debugName.data());
}

bool ComputePass::compileShader(
    wheels::ScopedScratch scopeAlloc, const Shader &shader)
{
    WHEELS_ASSERT(all(greaterThan(shader.groupSize, uvec3{0})));
    _groupSize = shader.groupSize;

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
        gDevice.logical().destroy(_shaderModule);

        ShaderReflection &reflection = compResult->reflection;

        _shaderModule = compResult->module;
        _shaderReflection = WHEELS_MOV(reflection);

        return true;
    }

    return false;
}
