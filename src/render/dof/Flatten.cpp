#include "Flatten.hpp"

#include "render/RenderResources.hpp"
#include "render/Utils.hpp"
#include "utils/Profiler.hpp"
#include "utils/Utils.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

using namespace glm;
using namespace wheels;

namespace render::dof
{

namespace
{

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/dof/flatten.comp",
        .debugName = String{alloc, "DepthOfFieldFlattenCS"},
        .groupSize =
            uvec3{Flatten::sFlattenFactor, Flatten::sFlattenFactor, 1u},
    };
}

} // namespace

void Flatten::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(WHEELS_MOV(scopeAlloc), shaderDefinitionCallback);

    m_initialized = true;
}

void Flatten::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

Flatten::Output Flatten::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    ImageHandle halfResCircleOfConfusion, const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  Flatten");

    Output ret;
    {
        const vk::Extent2D inputExtent = getExtent2D(halfResCircleOfConfusion);

        ret.tileMinMaxCircleOfConfusion = gRenderResources.images->create(
            gfx::ImageDescription{
                .format = vk::Format::eR16G16Sfloat,
                .width = roundedUpQuotient(
                    inputExtent.width, Flatten::sFlattenFactor),
                .height = roundedUpQuotient(
                    inputExtent.height, Flatten::sFlattenFactor),
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,
            },
            "tileMinMaxCircleOfConfusion");

        const vk::DescriptorSet storageSet = m_computePass.updateStorageSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(halfResCircleOfConfusion)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(ret.tileMinMaxCircleOfConfusion)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 2>{{
                    {halfResCircleOfConfusion,
                     gfx::ImageState::ComputeShaderRead},
                    {ret.tileMinMaxCircleOfConfusion,
                     gfx::ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "  Flatten");

        const uvec3 groupCount = m_computePass.groupCount(
            uvec3{inputExtent.width, inputExtent.height, 1u});
        m_computePass.record(cb, groupCount, Span{&storageSet, 1});
    }

    return ret;
}

} // namespace render::dof
