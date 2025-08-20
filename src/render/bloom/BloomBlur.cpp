#include "BloomBlur.hpp"

#include "render/RenderResources.hpp"
#include "render/RenderTargets.hpp"
#include "utils/Profiler.hpp"

#include <imgui.h>
#include <shader_structs/push_constants/bloom/blur.h>

using namespace glm;
using namespace wheels;

namespace
{

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/bloom/blur.comp",
        .debugName = String{alloc, "BloomBlurCS"},
    };
}

} // namespace

void BloomBlur::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(
        WHEELS_MOV(scopeAlloc), shaderDefinitionCallback,
        ComputePassOptions{
            .storageSetInstanceCount = 2,
        });

    m_initialized = true;
}

void BloomBlur::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

void BloomBlur::startFrame() { m_computePass.startFrame(); }

void BloomBlur::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, ImageHandle inOutHighlights,
    BloomResolutionScale resolutionScale, const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  Blur");

    {
        const Image &inputResource =
            gRenderResources.images->resource(inOutHighlights);
        const vk::Extent2D renderExtent = vk::Extent2D{
            .width = inputResource.extent.width,
            .height = inputResource.extent.height,
        };

        ImageHandle const pongImage = gRenderResources.images->create(
            ImageDescription{
                .format = sIlluminationFormat,
                .width = renderExtent.width,
                .height = renderExtent.height,
                .mipCount = inputResource.mipCount,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,

            },
            "BloomBlurPong");

        const Span<const vk::ImageView> inputViews =
            gRenderResources.images->subresourceViews(inOutHighlights);
        const Span<const vk::ImageView> pongViews =
            gRenderResources.images->subresourceViews(pongImage);
        WHEELS_ASSERT(inputViews.size() == 4);
        WHEELS_ASSERT(pongViews.size() == 4);

        StaticArray<vk::DescriptorImageInfo, 4> inputWriteInfos;
        StaticArray<vk::DescriptorImageInfo, 4> pongWriteInfos;
        for (uint32_t i = 0; i < 4; ++i)
        {
            inputWriteInfos[i] = vk::DescriptorImageInfo{
                .imageView = inputViews[i],
                .imageLayout = vk::ImageLayout::eGeneral,
            };
            pongWriteInfos[i] = vk::DescriptorImageInfo{
                .imageView = pongViews[i],
                .imageLayout = vk::ImageLayout::eGeneral,
            };
        }

        const vk::DescriptorSet pingSet = m_computePass.updateStorageSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(inOutHighlights).view,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                }},
                DescriptorInfo{pongWriteInfos},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler =
                        gRenderResources.bilinearBorderTransparentBlackSampler,
                }},
            }});
        const vk::DescriptorSet pongSet = m_computePass.updateStorageSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(pongImage).view,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                }},
                DescriptorInfo{inputWriteInfos},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler =
                        gRenderResources.bilinearBorderTransparentBlackSampler,
                }},
            }});

        PROFILER_GPU_SCOPE(cb, "  Blur");

        const uint32_t firstMipLevel =
            resolutionScale == BloomResolutionScale::Half ? 0 : 1;
        const uvec2 firstMipResolution =
            uvec2{renderExtent.width, renderExtent.height} /
            (resolutionScale == BloomResolutionScale::Half ? 1u : 2u);
        SinglePassData passData{
            .descriptorSet = pingSet,
            .mipLevel = firstMipLevel,
            .mipResolution = firstMipResolution,
            .transpose = false,
        };

        transition(
            scopeAlloc.child_scope(), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 2>{{
                    {inOutHighlights, ImageState::ComputeShaderSampledRead},
                    {pongImage, ImageState::ComputeShaderWrite},
                }},
            });

        // TODO:
        // This could be a single dispatch per direction instead of three
        recordSinglePass(cb, passData);
        passData.mipLevel++;
        passData.mipResolution /= 2;
        recordSinglePass(cb, passData);
        passData.mipLevel++;
        passData.mipResolution /= 2;
        recordSinglePass(cb, passData);

        passData.descriptorSet = pongSet;
        passData.mipLevel = firstMipLevel;
        passData.mipResolution = firstMipResolution;
        passData.transpose = true;

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 2>{{
                    {pongImage, ImageState::ComputeShaderSampledRead},
                    {inOutHighlights, ImageState::ComputeShaderWrite},
                }},
            });

        // TODO:
        // This could be a single dispatch per direction instead of three
        recordSinglePass(cb, passData);
        passData.mipLevel++;
        passData.mipResolution /= 2;
        recordSinglePass(cb, passData);
        passData.mipLevel++;
        passData.mipResolution /= 2;
        recordSinglePass(cb, passData);

        gRenderResources.images->release(pongImage);
    }
}

void BloomBlur::recordSinglePass(
    vk::CommandBuffer cb, const SinglePassData &data)
{
    WHEELS_ASSERT(all(greaterThan(data.mipResolution, uvec2(0))));

    const BlurPC pcBlock{
        .resolution = data.mipResolution,
        .invResolution = 1.f / vec2(data.mipResolution),
        .mipLevel = data.mipLevel,
        .transpose = data.transpose ? 1u : 0u,
    };
    const uvec3 groupCount =
        m_computePass.groupCount(uvec3{data.mipResolution, 1});
    m_computePass.record(cb, pcBlock, groupCount, Span{&data.descriptorSet, 1});
}
