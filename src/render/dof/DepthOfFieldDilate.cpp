#include "DepthOfFieldDilate.hpp"

#include "render/RenderResources.hpp"
#include "render/Utils.hpp"
#include "render/dof/DepthOfField.hpp"
#include "scene/Camera.hpp"
#include "utils/Profiler.hpp"
#include "utils/Utils.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

using namespace glm;
using namespace wheels;

namespace
{

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/dof/dilate.comp",
        .debugName = String{alloc, "DepthOfFieldDilateCS"},
    };
}

struct PcBlock
{
    ivec2 res{};
    vec2 invRes{};
    int32_t gatherRadius{1};
};

} // namespace

void DepthOfFieldDilate::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(WHEELS_MOV(scopeAlloc), shaderDefinitionCallback);

    m_initialized = true;
}

void DepthOfFieldDilate::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

DepthOfFieldDilate::Output DepthOfFieldDilate::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, ImageHandle tileMinMaxCoC,
    const Camera &cam, const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  Dilate");

    Output ret;
    {
        const vk::Extent2D inputExtent = getExtent2D(tileMinMaxCoC);

        ret.dilatedTileMinMaxCoC = gRenderResources.images->create(
            ImageDescription{
                .format = vk::Format::eR16G16Sfloat,
                .width = inputExtent.width,
                .height = inputExtent.height,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,
            },
            "dilatedTileMinMaxCoC");

        m_computePass.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(tileMinMaxCoC).view,
                    .imageLayout = vk::ImageLayout::eReadOnlyOptimal,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(ret.dilatedTileMinMaxCoC)
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
                    {tileMinMaxCoC, ImageState::ComputeShaderSampledRead},
                    {ret.dilatedTileMinMaxCoC, ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "  Dilate");

        const CameraParameters &camParams = cam.parameters();
        const float maxBgCoCInUnits =
            (camParams.apertureDiameter * camParams.focalLength) /
            (camParams.focusDistance - camParams.focalLength);

        // This is in the flattened tile resolution but it should cover the half
        // res CoC as it's calculated from an on-sensor size.
        const int32_t maxBgCoCInPixels = asserted_cast<int32_t>(std::ceil(
            (maxBgCoCInUnits / cam.sensorWidth()) *
            static_cast<float>(inputExtent.width)));
        // TODO:
        // This can be significantly larger than any actual CoC in the image.
        // Track maxCoC in a GPU buffer instead and use that?
        const int32_t gatherRadius = std::max(
            maxBgCoCInPixels * asserted_cast<int32_t>(
                                   std::ceil(DepthOfField::sMaxFgCoCFactor)),
            1);

        const PcBlock pcBlock{
            .res = ivec2(inputExtent.width, inputExtent.height),
            .invRes = 1.f / vec2(inputExtent.width, inputExtent.height),
            .gatherRadius = gatherRadius,
        };

        const uvec3 groupCount = m_computePass.groupCount(
            uvec3{inputExtent.width, inputExtent.height, 1u});
        const vk::DescriptorSet storageSet =
            m_computePass.storageSet(nextFrame);
        m_computePass.record(cb, pcBlock, groupCount, Span{&storageSet, 1});
    }

    return ret;
}
