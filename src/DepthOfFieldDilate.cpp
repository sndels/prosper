#include "DepthOfFieldDilate.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

vk::Extent2D getInputExtent(
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

DepthOfFieldDilate::DepthOfFieldDilate(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);
    assert(staticDescriptorsAlloc != nullptr);

    printf("Creating DepthOfFieldDilate\n");
    if (!compileShaders(scopeAlloc.child_scope()))
        throw std::runtime_error(
            "DepthOfFieldDilate shader compilation failed");

    createDescriptorSets(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    createPipeline();
}

DepthOfFieldDilate::~DepthOfFieldDilate()
{
    if (_device != nullptr)
    {
        destroyPipelines();

        _device->logical().destroy(_descriptorSetLayout);

        _device->logical().destroy(_compSM);
    }
}

void DepthOfFieldDilate::recompileShaders(wheels::ScopedScratch scopeAlloc)
{
    if (compileShaders(scopeAlloc.child_scope()))
    {
        destroyPipelines();
        createPipeline();
    }
}

bool DepthOfFieldDilate::compileShaders(ScopedScratch scopeAlloc)
{
    printf("Compiling DepthOfFieldDilate shaders\n");

    Optional<Device::ShaderCompileResult> compResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/dof/dilate.comp",
                                          .debugName = "DepthOfFieldDilateCS",
                                      });

    if (compResult.has_value())
    {
        _device->logical().destroy(_compSM);

        ShaderReflection &reflection = compResult->reflection;

        _compSM = compResult->module;
        _shaderReflection = WHEELS_MOV(reflection);

        return true;
    }

    return false;
}

DepthOfFieldDilate::Output DepthOfFieldDilate::record(
    vk::CommandBuffer cb, ImageHandle tileMinMaxCoC, const uint32_t nextFrame,
    Profiler *profiler)
{
    assert(profiler != nullptr);

    Output ret;
    {
        const vk::Extent2D inputExtent =
            getInputExtent(*_resources, tileMinMaxCoC);

        ret.dilatedTileMinMaxCoC = _resources->images.create(
            ImageDescription{
                .format = vk::Format::eR16G16Sfloat,
                .width = inputExtent.width,
                .height = inputExtent.height,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,
            },
            "dilatedTileMinMaxCoC");

        updateDescriptorSet(nextFrame, tileMinMaxCoC, ret);

        recordBarriers(cb, tileMinMaxCoC, ret);

        const auto _s = profiler->createCpuGpuScope(cb, "DepthOfFieldDilate");

        cb.bindPipeline(vk::PipelineBindPoint::eCompute, _pipeline);

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, // firstSet
            1, &_descriptorSets[nextFrame], 0, nullptr);

        const auto groups =
            (glm::uvec2{inputExtent.width, inputExtent.height} - 1u) / 16u + 1u;
        cb.dispatch(groups.x, groups.y, 1);
    }

    return ret;
}

void DepthOfFieldDilate::recordBarriers(
    vk::CommandBuffer cb, ImageHandle tileMinMaxCoC, const Output &output) const
{
    const StaticArray imageBarriers{
        _resources->images.transitionBarrier(
            tileMinMaxCoC,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderRead,
                .layout = vk::ImageLayout::eGeneral,
            }),
        _resources->images.transitionBarrier(
            output.dilatedTileMinMaxCoC,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderWrite,
                .layout = vk::ImageLayout::eGeneral,
            }),
    };

    cb.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount =
            asserted_cast<uint32_t>(imageBarriers.size()),
        .pImageMemoryBarriers = imageBarriers.data(),
    });
}

void DepthOfFieldDilate::destroyPipelines()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

void DepthOfFieldDilate::createDescriptorSets(
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

void DepthOfFieldDilate::updateDescriptorSet(
    uint32_t nextFrame, ImageHandle tileMinMaxCoC, const Output &output)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?

    const vk::DescriptorImageInfo inputInfo{
        .imageView = _resources->images.resource(tileMinMaxCoC).view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    const vk::DescriptorImageInfo outputInfo{
        .imageView =
            _resources->images.resource(output.dilatedTileMinMaxCoC).view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };

    const vk::DescriptorSet ds = _descriptorSets[nextFrame];

    assert(_shaderReflection.has_value());
    const StaticArray descriptorWrites =
        _shaderReflection->generateDescriptorWrites<2>(
            0, ds,
            {
                DescriptorInfo{inputInfo},
                DescriptorInfo{outputInfo},
            });

    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void DepthOfFieldDilate::createPipeline()
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
                .module = _compSM,
                .pName = "main",
            },
        .layout = _pipelineLayout,
    };

    _pipeline = createComputePipeline(
        _device->logical(), createInfo, "DepthOfFieldDilate");
}
