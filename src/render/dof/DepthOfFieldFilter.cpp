#include "DepthOfFieldFilter.hpp"

#include "../../utils/Profiler.hpp"
#include "../../utils/Utils.hpp"
#include "../RenderResources.hpp"
#include "../RenderTargets.hpp"

using namespace glm;
using namespace wheels;

namespace
{

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/dof/filter.comp",
        .debugName = String{alloc, "DepthOfFieldFilterCS"},
    };
}

} // namespace

void DepthOfFieldFilter::init(
    ScopedScratch scopeAlloc, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc)
{
    WHEELS_ASSERT(!_initialized);
    WHEELS_ASSERT(resources != nullptr);

    _resources = resources;
    _computePass.init(
        WHEELS_MOV(scopeAlloc), staticDescriptorsAlloc,
        shaderDefinitionCallback,
        ComputePassOptions{
            .perFrameRecordLimit = 2,
        });

    _initialized = true;
}

void DepthOfFieldFilter::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(_initialized);

    _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

void DepthOfFieldFilter::startFrame() { _computePass.startFrame(); }

DepthOfFieldFilter::Output DepthOfFieldFilter::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    ImageHandle inIlluminationWeight, const uint32_t nextFrame,
    const DebugNames &debugNames, Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);
    WHEELS_ASSERT(profiler != nullptr);

    const Image &inRes = _resources->images.resource(inIlluminationWeight);
    Output ret;
    // TODO:
    // Support querying description of a resource to create a new one from it?
    // Or have a dedicated create_matching()?
    ret.filteredIlluminationWeight = _resources->images.create(
        ImageDescription{
            .format = vk::Format::eR16G16B16A16Sfloat,
            .width = inRes.extent.width,
            .height = inRes.extent.height,
            .usageFlags = vk::ImageUsageFlagBits::eSampled |
                          vk::ImageUsageFlagBits::eStorage,
        },
        debugNames.outRes);

    _computePass.updateDescriptorSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{{
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = inRes.view,
                .imageLayout = vk::ImageLayout::eReadOnlyOptimal,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    _resources->images.resource(ret.filteredIlluminationWeight)
                        .view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .sampler = _resources->nearestSampler,
            }},
        }});

    transition(
        WHEELS_MOV(scopeAlloc), *_resources, cb,
        Transitions{
            .images = StaticArray<ImageTransition, 2>{{
                {inIlluminationWeight, ImageState::ComputeShaderSampledRead},
                {ret.filteredIlluminationWeight,
                 ImageState::ComputeShaderWrite},
            }},
        });

    const auto _s = profiler->createCpuGpuScope(cb, debugNames.scope);

    const vk::DescriptorSet descriptorSet = _computePass.storageSet(nextFrame);

    // Compute pass calculates group counts assuming extent / groupSize
    // TODO:
    // Rethink this interface. record() might be better off taking in group size
    // and there could be a groupSize(extent)-method that can be used to get the
    // typical calculation.
    const uvec3 extent{inRes.extent.width, inRes.extent.height, 1};
    _computePass.record(cb, extent, Span{&descriptorSet, 1});

    return ret;
}
