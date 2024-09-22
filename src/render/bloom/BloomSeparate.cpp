#include "BloomSeparate.hpp"

#include "render/RenderResources.hpp"
#include "render/RenderTargets.hpp"
#include "render/Utils.hpp"
#include "utils/Profiler.hpp"

#include <imgui.h>
#include <shader_structs/push_constants/bloom/separate.h>

using namespace glm;
using namespace wheels;

namespace
{

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/bloom/separate.comp",
        .debugName = String{alloc, "BloomSeparateCS"},
    };
}

} // namespace

void BloomSeparate::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(WHEELS_MOV(scopeAlloc), shaderDefinitionCallback);

    m_initialized = true;
}

void BloomSeparate::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

void BloomSeparate::drawUi()
{
    ImGui::SliderFloat("Threshold", &m_threshold, 0.f, 10.f);
}

BloomSeparate::Output BloomSeparate::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Input &input,
    const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  Separate");

    Output ret;
    {
        const vk::Extent2D renderExtent = getExtent2D(input.illumination);

        ret.highlights = gRenderResources.images->create(
            ImageDescription{
                .format = sIlluminationFormat,
                .width = renderExtent.width,
                .height = renderExtent.height,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,

            },
            "IlluminationHighlights");

        m_computePass.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(input.illumination)
                            .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(ret.highlights).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 2>{{
                    {input.illumination, ImageState::ComputeShaderRead},
                    {ret.highlights, ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "  Separate");

        const vk::DescriptorSet descriptorSet =
            m_computePass.storageSet(nextFrame);

        const SeparatePC pcBlock{
            .threshold = m_threshold,
        };
        const uvec3 groupCount = m_computePass.groupCount(
            uvec3{renderExtent.width, renderExtent.height, 1u});
        m_computePass.record(cb, pcBlock, groupCount, Span{&descriptorSet, 1});
    }

    return ret;
}
