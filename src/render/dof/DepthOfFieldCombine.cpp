#include "DepthOfFieldCombine.hpp"

#include "render/RenderResources.hpp"
#include "render/RenderTargets.hpp"
#include "render/Utils.hpp"
#include "utils/Profiler.hpp"

#include <imgui.h>

using namespace glm;
using namespace wheels;

namespace
{

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/dof/combine.comp",
        .debugName = String{alloc, "DepthOfFieldCombineCS"},
    };
}

} // namespace

void DepthOfFieldCombine::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(WHEELS_MOV(scopeAlloc), shaderDefinitionCallback);

    m_initialized = true;
}

void DepthOfFieldCombine::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

DepthOfFieldCombine::Output DepthOfFieldCombine::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Input &input,
    const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  Combine");

    Output ret;
    {
        const vk::Extent2D renderExtent = getExtent2D(input.illumination);

        ret.combinedIlluminationDoF =
            createIllumination(renderExtent, "CombinedIllumnationDoF");

        const StaticArray descriptorInfos{{
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images
                                 ->resource(input.halfResFgBokehWeight)
                                 .view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images
                                 ->resource(input.halfResBgBokehWeight)
                                 .view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images
                                 ->resource(input.halfResCircleOfConfusion)
                                 .view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(input.illumination).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images
                                 ->resource(ret.combinedIlluminationDoF)
                                 .view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
        }};
        m_computePass.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame, descriptorInfos);

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 5>{{
                    {input.halfResFgBokehWeight, ImageState::ComputeShaderRead},
                    {input.halfResBgBokehWeight, ImageState::ComputeShaderRead},
                    {input.halfResCircleOfConfusion,
                     ImageState::ComputeShaderRead},
                    {input.illumination, ImageState::ComputeShaderRead},
                    {ret.combinedIlluminationDoF,
                     ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "  Combine");

        const uvec3 groupCount = m_computePass.groupCount(
            uvec3{renderExtent.width, renderExtent.height, 1u});
        const vk::DescriptorSet storageSet =
            m_computePass.storageSet(nextFrame);
        m_computePass.record(cb, groupCount, Span{&storageSet, 1});
    }

    return ret;
}
