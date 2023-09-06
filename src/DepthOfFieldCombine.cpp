#include "DepthOfFieldCombine.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include "RenderTargets.hpp"
#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

vk::Extent2D getRenderExtent(
    const RenderResources &resources, ImageHandle illumination)
{
    const vk::Extent3D targetExtent =
        resources.images.resource(illumination).extent;
    assert(targetExtent.depth == 1);

    return vk::Extent2D{
        .width = targetExtent.width,
        .height = targetExtent.height,
    };
}

} // namespace

DepthOfFieldCombine::DepthOfFieldCombine(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);
    assert(staticDescriptorsAlloc != nullptr);

    printf("Creating DepthOfFieldCombine\n");
    if (!compileShaders(scopeAlloc.child_scope()))
        throw std::runtime_error(
            "DepthOfFieldCombine shader compilation failed");

    createDescriptorSets(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    createPipeline();
}

DepthOfFieldCombine::~DepthOfFieldCombine()
{
    if (_device != nullptr)
    {
        destroyPipelines();

        _device->logical().destroy(_descriptorSetLayout);

        _device->logical().destroy(_shaderModule);
    }
}

void DepthOfFieldCombine::recompileShaders(wheels::ScopedScratch scopeAlloc)
{
    if (compileShaders(scopeAlloc.child_scope()))
    {
        destroyPipelines();
        createPipeline();
    }
}

bool DepthOfFieldCombine::compileShaders(ScopedScratch scopeAlloc)
{
    printf("Compiling DepthOfFieldCombine shaders\n");

    Optional<Device::ShaderCompileResult> compResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/dof/combine.comp",
                                          .debugName = "DepthOfFieldCombineCS",
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

DepthOfFieldCombine::Output DepthOfFieldCombine::record(
    vk::CommandBuffer cb, const Input &input, const uint32_t nextFrame,
    Profiler *profiler)
{
    assert(profiler != nullptr);

    Output ret;
    {
        const vk::Extent2D renderExtent =
            getRenderExtent(*_resources, input.illumination);

        ret.combinedIlluminationDoF = createIllumination(
            *_resources, renderExtent, "CombinedIllumnationDoF");

        updateDescriptorSet(nextFrame, input, ret);

        recordBarriers(cb, input, ret);

        const auto _s = profiler->createCpuGpuScope(cb, "DepthOfFieldCombine");

        cb.bindPipeline(vk::PipelineBindPoint::eCompute, _pipeline);

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, // firstSet
            1, &_descriptorSets[nextFrame], 0, nullptr);

        const auto groups =
            (glm::uvec2{renderExtent.width, renderExtent.height} - 1u) / 16u +
            1u;
        cb.dispatch(groups.x, groups.y, 1);
    }

    return ret;
}

void DepthOfFieldCombine::recordBarriers(
    vk::CommandBuffer cb, const Input &input, const Output &output) const
{
    transition<5>(
        *_resources, cb,
        {
            {input.halfResFgBokehWeight, ImageState::ComputeShaderRead},
            {input.halfResBgBokehWeight, ImageState::ComputeShaderRead},
            {input.halfResCircleOfConfusion, ImageState::ComputeShaderRead},
            {input.illumination, ImageState::ComputeShaderRead},
            {output.combinedIlluminationDoF, ImageState::ComputeShaderWrite},
        });
}

void DepthOfFieldCombine::destroyPipelines()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

void DepthOfFieldCombine::createDescriptorSets(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    assert(_shaderReflection.has_value());
    const Array<vk::DescriptorSetLayoutBinding> layoutBindings =
        _shaderReflection->generateLayoutBindings(
            scopeAlloc, 0, vk::ShaderStageFlagBits::eCompute);

    _descriptorSetLayout = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
            .pBindings = layoutBindings.data(),
        });

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        _descriptorSetLayout};
    staticDescriptorsAlloc->allocate(layouts, _descriptorSets);
}

void DepthOfFieldCombine::updateDescriptorSet(
    uint32_t nextFrame, const Input &input, const Output &output)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?

    const StaticArray bindingInfos = {
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView =
                _resources->images.resource(input.halfResFgBokehWeight).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView =
                _resources->images.resource(input.halfResBgBokehWeight).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},

        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView =
                _resources->images.resource(input.halfResCircleOfConfusion)
                    .view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView = _resources->images.resource(input.illumination).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView =
                _resources->images.resource(output.combinedIlluminationDoF)
                    .view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
    };

    const vk::DescriptorSet ds = _descriptorSets[nextFrame];

    assert(_shaderReflection.has_value());
    const StaticArray descriptorWrites =
        _shaderReflection->generateDescriptorWrites(0, ds, bindingInfos);

    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void DepthOfFieldCombine::createPipeline()
{
    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &_descriptorSetLayout,
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

    _pipeline = createComputePipeline(
        _device->logical(), createInfo, "DepthOfFieldCombine");
}
