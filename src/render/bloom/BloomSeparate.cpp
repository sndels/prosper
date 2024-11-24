#include "BloomSeparate.hpp"

#include "render/RenderResources.hpp"
#include "render/RenderTargets.hpp"
#include "render/Utils.hpp"
#include "render/bloom/BloomFft.hpp"
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

ImageHandle BloomSeparate::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Input &input,
    const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  Separate");

    ImageHandle ret;
    {
        const vk::Extent2D inputExtent = getExtent2D(input.illumination);

        const uint32_t dim = std::max(
            std::bit_ceil(std::max(inputExtent.width, inputExtent.height)) / 2,
            BloomFft::sMinResolution);

        ret = gRenderResources.images->create(
            ImageDescription{
                .format = sIlluminationFormat,
                .width = dim,
                .height = dim,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,

            },
            "BloomFftPingPing");

        m_computePass.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(input.illumination)
                            .view,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images->resource(ret).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler = gRenderResources.bilinearSampler,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 2>{{
                    {input.illumination, ImageState::ComputeShaderSampledRead},
                    {ret, ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "  Separate");

        const vk::DescriptorSet descriptorSet =
            m_computePass.storageSet(nextFrame);

        const SeparatePC pcBlock{
            .invInResolution = 1.f /
                               vec2{
                                   static_cast<float>(inputExtent.width),
                                   static_cast<float>(inputExtent.height),
                               },
            .threshold = m_threshold,
        };
        const uvec3 groupCount = m_computePass.groupCount(uvec3{dim, dim, 1u});
        m_computePass.record(cb, pcBlock, groupCount, Span{&descriptorSet, 1});
    }

    return ret;
}
