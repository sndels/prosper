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

ComputePass::ComputePass(
    wheels::ScopedScratch scopeAlloc, Device *device,
    DescriptorAllocator *staticDescriptorsAlloc,
    const std::function<Shader(wheels::Allocator &)> &shaderDefinitionCallback,
    const ComputePassOptions &options)
: _device{device}
, _storageSetIndex{options.storageSetIndex}
{
    WHEELS_ASSERT(staticDescriptorsAlloc != nullptr);
    WHEELS_ASSERT(
        (_storageSetIndex == options.externalDsLayouts.size()) &&
        "Implementation assumes that the pass storage set is the last set and "
        "is placed right after the last external one");

    printf("Creating ComputePass\n");
    if (!compileShader(scopeAlloc.child_scope(), shaderDefinitionCallback))
        throw std::runtime_error("Shader compilation failed");

    createDescriptorSets(
        scopeAlloc.child_scope(), staticDescriptorsAlloc,
        options.storageStageFlags);
    createPipeline(scopeAlloc.child_scope(), options.externalDsLayouts);
}

ComputePass::~ComputePass()
{
    if (_device != nullptr)
    {
        destroyPipelines();

        _device->logical().destroy(_storageSetLayout);

        _device->logical().destroy(_shaderModule);
    }
}

bool ComputePass::recompileShader(
    wheels::ScopedScratch scopeAlloc,
    const wheels::HashSet<std::filesystem::path> &changedFiles,
    const std::function<Shader(wheels::Allocator &)> &shaderDefinitionCallback,
    wheels::Span<const vk::DescriptorSetLayout> externalDsLayouts)
{
    WHEELS_ASSERT(_shaderReflection.has_value());
    if (!_shaderReflection->affected(changedFiles))
        return false;

    if (compileShader(scopeAlloc.child_scope(), shaderDefinitionCallback))
    {
        destroyPipelines();
        createPipeline(scopeAlloc.child_scope(), externalDsLayouts);
        return true;
    }
    return false;
}

void ComputePass::updateDescriptorSet(
    ScopedScratch scopeAlloc, uint32_t nextFrame,
    Span<const DescriptorInfo> descriptorInfos)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?
    const vk::DescriptorSet ds = _storageSets[nextFrame];

    WHEELS_ASSERT(_shaderReflection.has_value());
    const wheels::Array descriptorWrites =
        _shaderReflection->generateDescriptorWrites(
            scopeAlloc, _storageSetIndex, ds, descriptorInfos);

    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

vk::DescriptorSet ComputePass::storageSet(uint32_t nextFrame) const
{
    return _storageSets[nextFrame];
}

vk::DescriptorSetLayout ComputePass::storageSetLayout() const
{
    return _storageSetLayout;
}

void ComputePass::record(
    vk::CommandBuffer cb, const uvec3 &extent,
    Span<const vk::DescriptorSet> descriptorSets,
    wheels::Span<const uint32_t> dynamicOffsets) const
{
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
}

void ComputePass::record(
    vk::CommandBuffer cb, Span<const uint8_t> pcBlockBytes, const uvec3 &extent,
    Span<const vk::DescriptorSet> descriptorSets,
    Span<const uint32_t> dynamicOffsets) const
{
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
}

void ComputePass::destroyPipelines()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

void ComputePass::createDescriptorSets(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc,
    vk::ShaderStageFlags storageStageFlags)
{
    WHEELS_ASSERT(_shaderReflection.has_value());
    _storageSetLayout = _shaderReflection->createDescriptorSetLayout(
        WHEELS_MOV(scopeAlloc), *_device, _storageSetIndex, storageStageFlags);

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        _storageSetLayout};
    staticDescriptorsAlloc->allocate(layouts, _storageSets);
}

void ComputePass::createPipeline(
    wheels::ScopedScratch scopeAlloc,
    Span<const vk::DescriptorSetLayout> externalDsLayouts)
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
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
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
        createComputePipeline(_device->logical(), createInfo, "ComputePass");
}

bool ComputePass::compileShader(
    wheels::ScopedScratch scopeAlloc,
    const std::function<Shader(wheels::Allocator &)> &shaderDefinitionCallback)
{
    Shader shader = shaderDefinitionCallback(scopeAlloc);
    WHEELS_ASSERT(all(greaterThan(shader.groupSize, uvec3{0})));
    _groupSize = shader.groupSize;

    printf("Compiling %s\n", shader.debugName.c_str());

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
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = shader.relPath,
                                          .debugName = shader.debugName.c_str(),
                                          .defines = defines,
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
