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
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(
        WHEELS_MOV(scopeAlloc), staticDescriptorsAlloc,
        shaderDefinitionCallback,
        ComputePassOptions{
            .perFrameRecordLimit = 2,
        });

    m_initialized = true;
}

void DepthOfFieldFilter::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

void DepthOfFieldFilter::startFrame() { m_computePass.startFrame(); }

DepthOfFieldFilter::Output DepthOfFieldFilter::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    ImageHandle inIlluminationWeight, const uint32_t nextFrame,
    const DebugNames &debugNames)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE(debugNames.scope);

    const Image &inRes =
        gRenderResources.images->resource(inIlluminationWeight);
    Output ret;
    // TODO:
    // Support querying description of a resource to create a new one from it?
    // Or have a dedicated create_matching()?
    ret.filteredIlluminationWeight = gRenderResources.images->create(
        ImageDescription{
            .format = vk::Format::eR16G16B16A16Sfloat,
            .width = inRes.extent.width,
            .height = inRes.extent.height,
            .usageFlags = vk::ImageUsageFlagBits::eSampled |
                          vk::ImageUsageFlagBits::eStorage,
        },
        debugNames.outRes);

    m_computePass.updateDescriptorSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{{
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = inRes.view,
                .imageLayout = vk::ImageLayout::eReadOnlyOptimal,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images
                                 ->resource(ret.filteredIlluminationWeight)
                                 .view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .sampler = gRenderResources.nearestSampler,
            }},
        }});

    transition(
        WHEELS_MOV(scopeAlloc), cb,
        Transitions{
            .images = StaticArray<ImageTransition, 2>{{
                {inIlluminationWeight, ImageState::ComputeShaderSampledRead},
                {ret.filteredIlluminationWeight,
                 ImageState::ComputeShaderWrite},
            }},
        });

    PROFILER_GPU_SCOPE(cb, debugNames.scope);

    const vk::DescriptorSet descriptorSet = m_computePass.storageSet(nextFrame);

    const uvec3 groupCount = m_computePass.groupCount(
        uvec3{inRes.extent.width, inRes.extent.height, 1});
    m_computePass.record(cb, groupCount, Span{&descriptorSet, 1});

    return ret;
}
