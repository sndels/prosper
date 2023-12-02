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

ComputePass::~ComputePass()
{
    if (_device != nullptr)
    {
        destroyPipelines();

        _device->logical().destroy(_storageSetLayout);

        _device->logical().destroy(_shaderModule);
    }
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
    vk::CommandBuffer cb, const uvec3 &groups,
    Span<const vk::DescriptorSet> descriptorSets,
    wheels::Span<const uint32_t> dynamicOffsets) const
{
    WHEELS_ASSERT(all(greaterThan(groups, glm::uvec3{0u})));

    cb.bindPipeline(vk::PipelineBindPoint::eCompute, _pipeline);

    cb.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, _pipelineLayout,
        0, // firstSet
        asserted_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
        asserted_cast<uint32_t>(dynamicOffsets.size()), dynamicOffsets.data());

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
