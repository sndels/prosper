#include "DepthOfFieldSetup.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include "RenderTargets.hpp"
#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

enum BindingSet : uint32_t
{
    CameraBindingSet = 0,
    StorageBindingSet = 1,
    BindingSetCount = 2,
};

struct PCBlock
{
    float focusDistance{0.f};
    float maxBackgroundCoC{0.f};
};

vk::Extent2D getRenderExtent(
    const RenderResources &resources, ImageHandle illumination)
{
    const vk::Extent3D targetExtent =
        resources.images.resource(illumination).extent;
    assert(targetExtent.depth == 1);

    return vk::Extent2D{
        .width = targetExtent.width / 2,
        .height = targetExtent.height / 2,
    };
}

} // namespace

DepthOfFieldSetup::DepthOfFieldSetup(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    vk::DescriptorSetLayout camDsLayout)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);
    assert(staticDescriptorsAlloc != nullptr);

    printf("Creating DepthOfFieldSetup\n");
    if (!compileShaders(scopeAlloc.child_scope()))
        throw std::runtime_error("DepthOfFieldSetup shader compilation failed");

    createDescriptorSets(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    createPipeline(camDsLayout);
}

DepthOfFieldSetup::~DepthOfFieldSetup()
{
    if (_device != nullptr)
    {
        destroyPipelines();

        _device->logical().destroy(_descriptorSetLayout);

        _device->logical().destroy(_compSM);
    }
}

void DepthOfFieldSetup::recompileShaders(
    wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDsLayout)
{
    if (compileShaders(scopeAlloc.child_scope()))
    {
        destroyPipelines();
        createPipeline(camDsLayout);
    }
}

bool DepthOfFieldSetup::compileShaders(ScopedScratch scopeAlloc)
{
    printf("Compiling DepthOfFieldSetup shaders\n");

    String defines{scopeAlloc, 256};
    appendDefineStr(defines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(defines, "STORAGE_SET", StorageBindingSet);

    Optional<Device::ShaderCompileResult> compResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/dof/setup.comp",
                                          .debugName = "DepthOfFieldSetupCS",
                                          .defines = defines,
                                      });

    if (compResult.has_value())
    {
        _device->logical().destroy(_compSM);

        ShaderReflection &reflection = compResult->reflection;
        assert(sizeof(PCBlock) == reflection.pushConstantsBytesize());

        _compSM = compResult->module;
        _shaderReflection = WHEELS_MOV(reflection);

        return true;
    }

    return false;
}

DepthOfFieldSetup::Output DepthOfFieldSetup::record(
    vk::CommandBuffer cb, const Camera &cam, const Input &input,
    const uint32_t nextFrame, Profiler *profiler)
{
    assert(profiler != nullptr);

    Output ret;
    {
        const vk::Extent2D renderExtent =
            getRenderExtent(*_resources, input.illumination);

        ret.halfResIllumination = createIllumination(
            *_resources, renderExtent, "HalfResIllumination");
        ret.halfResCircleOfConfusion = _resources->images.create(
            ImageDescription{
                .format = vk::Format::eR16Sfloat,
                .width = renderExtent.width,
                .height = renderExtent.height,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,
            },
            "HalfResCircleOfConfusion");

        updateDescriptorSet(nextFrame, input, ret);

        recordBarriers(cb, input, ret);

        const auto _s = profiler->createCpuGpuScope(cb, "DepthOfFieldSetup");

        cb.bindPipeline(vk::PipelineBindPoint::eCompute, _pipeline);

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[CameraBindingSet] = cam.descriptorSet(nextFrame);
        descriptorSets[StorageBindingSet] = _descriptorSets[nextFrame];

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, // firstSet
            asserted_cast<uint32_t>(descriptorSets.capacity()),
            descriptorSets.data(), 0, nullptr);

        const float maxBgCoCInUnits =
            (cam.apertureDiameter() * cam.focalLength()) /
            (cam.focusDistance() - cam.focalLength());

        const float maxBgCoCInHalfResPixels =
            (maxBgCoCInUnits / cam.sensorWidth()) * renderExtent.width;

        const PCBlock pcBlock{
            .focusDistance = cam.focusDistance(),
            .maxBackgroundCoC = maxBgCoCInHalfResPixels,
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

void DepthOfFieldSetup::recordBarriers(
    vk::CommandBuffer cb, const Input &input, const Output &output) const
{
    const StaticArray imageBarriers{
        _resources->images.transitionBarrier(
            input.illumination, ImageState::ComputeShaderRead),
        _resources->images.transitionBarrier(
            input.depth, ImageState::ComputeShaderRead),
        _resources->images.transitionBarrier(
            output.halfResIllumination, ImageState::ComputeShaderWrite),
        _resources->images.transitionBarrier(
            output.halfResCircleOfConfusion, ImageState::ComputeShaderWrite),
    };

    cb.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount =
            asserted_cast<uint32_t>(imageBarriers.size()),
        .pImageMemoryBarriers = imageBarriers.data(),
    });
}

void DepthOfFieldSetup::destroyPipelines()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

void DepthOfFieldSetup::createDescriptorSets(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    assert(_shaderReflection.has_value());
    const Array<vk::DescriptorSetLayoutBinding> layoutBindings =
        _shaderReflection->generateLayoutBindings(
            scopeAlloc, StorageBindingSet, vk::ShaderStageFlagBits::eCompute);

    _descriptorSetLayout = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
            .pBindings = layoutBindings.data(),
        });

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        _descriptorSetLayout};
    staticDescriptorsAlloc->allocate(layouts, _descriptorSets);
}

void DepthOfFieldSetup::updateDescriptorSet(
    uint32_t nextFrame, const Input &input, const Output &output)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?

    const vk::DescriptorImageInfo illuminationInfo{
        .imageView = _resources->images.resource(input.illumination).view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    const vk::DescriptorImageInfo depthInfo{
        .imageView = _resources->images.resource(input.depth).view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    const vk::DescriptorImageInfo halfResIlluminationInfo{
        .imageView =
            _resources->images.resource(output.halfResIllumination).view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    const vk::DescriptorImageInfo halfResCircleOfConfusionInfo{
        .imageView =
            _resources->images.resource(output.halfResCircleOfConfusion).view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    const vk::DescriptorImageInfo depthSamplerInfo{
        .sampler = _resources->nearestSampler,
    };

    const vk::DescriptorSet ds = _descriptorSets[nextFrame];

    assert(_shaderReflection.has_value());
    const StaticArray descriptorWrites =
        _shaderReflection->generateDescriptorWrites<5>(
            StorageBindingSet, ds,
            {
                DescriptorInfo{illuminationInfo},
                DescriptorInfo{depthInfo},
                DescriptorInfo{halfResIlluminationInfo},
                DescriptorInfo{halfResCircleOfConfusionInfo},
                DescriptorInfo{depthSamplerInfo},
            });

    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void DepthOfFieldSetup::createPipeline(vk::DescriptorSetLayout camDsLayout)
{
    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(PCBlock),
    };

    StaticArray<vk::DescriptorSetLayout, BindingSetCount> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[CameraBindingSet] = camDsLayout;
    setLayouts[StorageBindingSet] = _descriptorSetLayout;

    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = asserted_cast<uint32_t>(setLayouts.capacity()),
            .pSetLayouts = setLayouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange,
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
        _device->logical(), createInfo, "DepthOfFieldSetup");
}
