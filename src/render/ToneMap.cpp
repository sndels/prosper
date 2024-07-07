#include "ToneMap.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include <fstream>

#include "../gfx/VkUtils.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/Utils.hpp"
#include "RenderResources.hpp"
#include "Utils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

struct PCBlock
{
    float exposure{1.f};
    float contrast{1.f};
};

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/tone_map.comp",
        .debugName = String{alloc, "ToneMapCS"},
    };
}

} // namespace

void ToneMap::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(scopeAlloc.child_scope(), shaderDefinitionCallback);

    m_lut.init(
        WHEELS_MOV(scopeAlloc), resPath("texture/tony_mc_mapface.dds"),
        ImageState::ComputeShaderSampledRead);

    m_initialized = true;
}

void ToneMap::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

void ToneMap::drawUi()
{
    WHEELS_ASSERT(m_initialized);

    ImGui::DragFloat("Exposure", &m_exposure, 0.01f, 0.001f, 10000.f);
    ImGui::DragFloat("Contrast", &m_contrast, 0.01f, 0.001f, 10000.f);
}

ToneMap::Output ToneMap::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, ImageHandle inColor,
    const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("ToneMap");

    Output ret;
    {
        const vk::Extent2D renderExtent = getExtent2D(inColor);

        ret = Output{
            .toneMapped = gRenderResources.images->create(
                ImageDescription{
                    .format = vk::Format::eR8G8B8A8Unorm,
                    .width = renderExtent.width,
                    .height = renderExtent.height,
                    .usageFlags =
                        vk::ImageUsageFlagBits::eSampled |         // Debug
                        vk::ImageUsageFlagBits::eStorage |         // ToneMap
                        vk::ImageUsageFlagBits::eColorAttachment | // ImGui
                        vk::ImageUsageFlagBits::eTransferSrc, // Blit to swap
                                                              // image
                },
                "toneMapped"),
        };

        const StaticArray descriptorInfos{{
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images->resource(inColor).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{m_lut.imageInfo()},
            DescriptorInfo{vk::DescriptorImageInfo{
                .sampler = gRenderResources.bilinearSampler,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(ret.toneMapped).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
        }};
        m_computePass.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame, descriptorInfos);

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 2>{{
                    {inColor, ImageState::ComputeShaderRead},
                    {ret.toneMapped, ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "ToneMap");

        const uvec3 groupCount = m_computePass.groupCount(
            uvec3{renderExtent.width, renderExtent.height, 1u});

        const vk::DescriptorSet storageSet =
            m_computePass.storageSet(nextFrame);
        m_computePass.record(
            cb,
            PCBlock{
                .exposure = m_exposure,
                .contrast = m_contrast,
            },
            groupCount, Span{&storageSet, 1});
    }

    return ret;
}
