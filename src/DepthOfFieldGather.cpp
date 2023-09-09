#include "DepthOfFieldGather.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include "RenderTargets.hpp"
#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

struct PCBlock
{
    uint frameIndex{0};
};

vk::Extent2D getRenderExtent(
    const RenderResources &resources, ImageHandle halfResIllumination)
{
    const vk::Extent3D targetExtent =
        resources.images.resource(halfResIllumination).extent;
    assert(targetExtent.depth == 1);

    return vk::Extent2D{
        .width = targetExtent.width,
        .height = targetExtent.height,
    };
}

} // namespace

DepthOfFieldGather::DepthOfFieldGather(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);
    assert(staticDescriptorsAlloc != nullptr);

    printf("Creating DepthOfFieldGather\n");
    if (!compileShaders(scopeAlloc.child_scope()))
        throw std::runtime_error(
            "DepthOfFieldGather shader compilation failed");

    createDescriptorSets(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    createPipeline();
}

DepthOfFieldGather::~DepthOfFieldGather()
{
    if (_device != nullptr)
    {
        destroyPipelines();

        _device->logical().destroy(_descriptorSetLayout);

        for (const vk::ShaderModule sm : _shaderModules)
            _device->logical().destroy(sm);
    }
}

void DepthOfFieldGather::recompileShaders(wheels::ScopedScratch scopeAlloc)
{
    if (compileShaders(scopeAlloc.child_scope()))
    {
        destroyPipelines();
        createPipeline();
    }
}

bool DepthOfFieldGather::compileShaders(ScopedScratch scopeAlloc)
{
    printf("Compiling DepthOfFieldGather shaders\n");

    auto compileShader = [&](GatherType gatherType)
    {
        assert(gatherType < GatherType_Count);

        String defines{scopeAlloc, 256};
        if (gatherType == GatherType_Background)
            appendDefineStr(defines, "GATHER_BACKGROUND");

        Optional<Device::ShaderCompileResult> compResult =
            _device->compileShaderModule(
                scopeAlloc.child_scope(),
                Device::CompileShaderModuleArgs{
                    .relPath = "shader/dof/gather.comp",
                    .debugName = gatherType == GatherType_Background
                                     ? "DepthOfFieldGatherBgCS"
                                     : "DepthOfFieldGatherFgCS",
                    .defines = defines,
                });

        if (compResult.has_value())
        {
            _device->logical().destroy(_shaderModules[gatherType]);

            ShaderReflection &reflection = compResult->reflection;
            assert(sizeof(PCBlock) == reflection.pushConstantsBytesize());

            _shaderModules[gatherType] = compResult->module;

            if (_shaderReflections.size() > gatherType)
                _shaderReflections[gatherType] = WHEELS_MOV(reflection);
            else
                _shaderReflections.push_back(WHEELS_MOV(reflection));

            return true;
        }
        return false;
    };

    for (uint32_t i = 0; i < GatherType_Count; ++i)
        if (!compileShader(static_cast<GatherType>(i)))
            return false;

    return true;
}

DepthOfFieldGather::Output DepthOfFieldGather::record(
    vk::CommandBuffer cb, const Input &input, GatherType gatherType,
    const uint32_t nextFrame, Profiler *profiler)
{
    assert(profiler != nullptr);
    assert(gatherType < GatherType_Count);

    if (gatherType == GatherType_Foreground)
        _frameIndex = (_frameIndex + 1) % 128;

    Output ret;
    {
        const vk::Extent2D renderExtent =
            getRenderExtent(*_resources, input.halfResIllumination);

        ret.halfResBokehColorWeight = _resources->images.create(
            ImageDescription{
                .format = vk::Format::eR16G16B16A16Sfloat,
                .width = renderExtent.width,
                .height = renderExtent.height,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,
            },
            gatherType == GatherType_Background ? "halfResBgBokehColorWeight"
                                                : "halfResFgBokehColorWeight");
        vk::DescriptorSet const descriptorSet =
            _descriptorSets[GatherType_Count * nextFrame + gatherType];

        updateDescriptorSet(descriptorSet, gatherType, input, ret);

        recordBarriers(cb, input, ret);

        const auto _s = profiler->createCpuGpuScope(
            cb, gatherType == GatherType_Background ? "DepthOfFieldGatherBg"
                                                    : "DepthOfFieldGatherFg");

        cb.bindPipeline(
            vk::PipelineBindPoint::eCompute, _pipelines[gatherType]);

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, // firstSet
            1, &descriptorSet, 0, nullptr);

        const PCBlock pcBlock{
            .frameIndex = _frameIndex,
        };
        cb.pushConstants(
            _pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
            sizeof(PCBlock), &pcBlock);

        const auto groups =
            (glm::uvec2{renderExtent.width, renderExtent.height} - 1u) / 16u +
            1u;
        cb.dispatch(groups.x, groups.y, 1);
    }

    return ret;
}

void DepthOfFieldGather::recordBarriers(
    vk::CommandBuffer cb, const Input &input, const Output &output) const
{
    transition<4>(
        *_resources, cb,
        {
            {input.halfResIllumination, ImageState::ComputeShaderRead},
            {input.halfResCoC, ImageState::ComputeShaderRead},
            {input.dilatedTileMinMaxCoC, ImageState::ComputeShaderRead},
            {output.halfResBokehColorWeight, ImageState::ComputeShaderWrite},
        });
}

void DepthOfFieldGather::destroyPipelines()
{
    for (const vk::Pipeline p : _pipelines)
        _device->logical().destroy(p);
    _pipelines.clear();
    _device->logical().destroy(_pipelineLayout);
}

void DepthOfFieldGather::createDescriptorSets(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    assert(_shaderReflections.size() == 2);
    const Array<vk::DescriptorSetLayoutBinding> layoutBindings =
        _shaderReflections[0].generateLayoutBindings(
            scopeAlloc, 0, vk::ShaderStageFlagBits::eCompute);

    _descriptorSetLayout = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
            .pBindings = layoutBindings.data(),
        });

#ifndef NDEBUG
    // Make sure we really can use the same layout for all sets
    // This is maybe overkill
    for (uint32_t gatherType = 0; gatherType < GatherType_Count; ++gatherType)
    {
        const Array<vk::DescriptorSetLayoutBinding> lb =
            _shaderReflections[gatherType].generateLayoutBindings(
                scopeAlloc, 0, vk::ShaderStageFlagBits::eCompute);
        assert(lb.size() == layoutBindings.size());
        for (size_t i = 0; i < lb.size(); ++i)
            assert(lb[i] == layoutBindings[i]);
    }
#endif // NDEBUG

    const StaticArray<vk::DescriptorSetLayout, DescriptorSets::capacity()>
        layouts{_descriptorSetLayout};
    staticDescriptorsAlloc->allocate(layouts, _descriptorSets);
}

void DepthOfFieldGather::updateDescriptorSet(
    vk::DescriptorSet descriptorSet, GatherType gatherType, const Input &input,
    const Output &output)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?

    const StaticArray bindingInfos = {
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView =
                _resources->images.resource(input.halfResIllumination).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView = _resources->images.resource(input.halfResCoC).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView =
                _resources->images.resource(input.dilatedTileMinMaxCoC).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView =
                _resources->images.resource(output.halfResBokehColorWeight)
                    .view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
    };

    assert(_shaderReflections.size() > gatherType);
    const StaticArray descriptorWrites =
        _shaderReflections[gatherType].generateDescriptorWrites(
            0, descriptorSet, bindingInfos);

    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void DepthOfFieldGather::createPipeline()
{
    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(PCBlock),
    };
    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &_descriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange,
        });

    for (uint32_t i = 0; i < GatherType_Count; ++i)
    {
        const vk::ComputePipelineCreateInfo createInfo{
            .stage =
                {
                    .stage = vk::ShaderStageFlagBits::eCompute,
                    .module = _shaderModules[i],
                    .pName = "main",
                },
            .layout = _pipelineLayout,
        };

        _pipelines.push_back(createComputePipeline(
            _device->logical(), createInfo, "DepthOfFieldGather"));
    }
}
