#include "BloomFft.hpp"

#include "render/RenderResources.hpp"
#include "render/Utils.hpp"
#include "utils/Profiler.hpp"

#include <bit>
#include <imgui.h>
#include <shader_structs/push_constants/bloom/fft.h>

using namespace glm;
using namespace wheels;

namespace
{

const vk::Format sFftFormat = vk::Format::eR32G32B32A32Sfloat;

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/bloom/fft.comp",
        .debugName = String{alloc, "BloomFftCS"},
        .groupSize = {64, 1, 1},
    };
}

} // namespace

void BloomFft::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(
        WHEELS_MOV(scopeAlloc), shaderDefinitionCallback,
        ComputePassOptions{
            // 7 passes can transform two components of 16k x 16k by rows
            // Twice that for four components
            // Twice that by columns
            // Twice that for inverse transform
            .perFrameRecordLimit = 7 * 2 * 2 * 2,
        });

    m_initialized = true;
}

void BloomFft::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

void BloomFft::startFrame() { m_computePass.startFrame(); }

ComplexImagePair BloomFft::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    const ComplexImagePair &input, const uint32_t nextFrame, bool inverse)
{
    WHEELS_ASSERT(m_initialized);

    // TODO:
    // - Radix4
    // - Shared memory version
    // - Two components at a time
    // - Two passes over the data for four components
    // - Leverage symmetry of input imaginary==0 and output imaginary being
    // discarded
    //   - Should be able to store just half of the specturm
    //   - Need different fft/ifft implementations, can't just flip real<->imag
    //     - first fft/last ifft are unique at least
    // - Compare to DIT Cooley-Tukey
    //   - Ryg makes a convicing argument for that, also some FMA optimizations
    //     https://fgiesen.wordpress.com/2023/03/19/notes-on-ffts-for-implementers/

    PROFILER_CPU_GPU_SCOPE(cb, inverse ? "  InverseFft" : "  Fft");

    const vk::Extent2D inputExtent = getExtent2D(input.real);
    const uint32_t outputDim =
        std::bit_ceil(std::max(inputExtent.width, inputExtent.height));
    const vk::Extent2D outputExtent{
        .width = outputDim,
        .height = outputDim,
    };

    ImageHandle const pingReal = gRenderResources.images->create(
        ImageDescription{
            .format = sFftFormat,
            .width = outputExtent.width,
            .height = outputExtent.height,
            .usageFlags = vk::ImageUsageFlagBits::eSampled |
                          vk::ImageUsageFlagBits::eStorage,
        },
        inverse ? "BloomInvFftPingReal" : "BloomFftPingReal");
    ImageHandle const pingImag = gRenderResources.images->create(
        ImageDescription{
            .format = sFftFormat,
            .width = outputExtent.width,
            .height = outputExtent.height,
            .usageFlags = vk::ImageUsageFlagBits::eSampled |
                          vk::ImageUsageFlagBits::eStorage,
        },
        inverse ? "BloomInvFftPingImag" : "BloomFftPingImag");
    ImageHandle const pongReal = gRenderResources.images->create(
        ImageDescription{
            .format = sFftFormat,
            .width = outputExtent.width,
            .height = outputExtent.height,
            .usageFlags = vk::ImageUsageFlagBits::eSampled |
                          vk::ImageUsageFlagBits::eStorage,
        },
        inverse ? "BloomInvFftPongReal" : "BloomFftPongReal");
    ImageHandle const pongImag = gRenderResources.images->create(
        ImageDescription{
            .format = sFftFormat,
            .width = outputExtent.width,
            .height = outputExtent.height,
            .usageFlags = vk::ImageUsageFlagBits::eSampled |
                          vk::ImageUsageFlagBits::eStorage,
        },
        inverse ? "BloomInvFftPongImag" : "BloomFftPongImag");

    // Rows first
    IterationData iterData{
        .input =
            ComplexImagePair{
                .real = inverse ? input.imag : input.real,
                .imag = inverse ? input.real : input.imag,
            },
        .output =
            ComplexImagePair{
                .real = pingReal,
                .imag = pingImag,
            },
        .ns = 1,
        .transpose = inverse,
    };
    iterData.r = 2;
    // TODO: Start with  R=4 if divides evenly already?
    doIteration(scopeAlloc.child_scope(), cb, iterData, nextFrame);
    iterData.input = ComplexImagePair{
        .real = pingReal,
        .imag = pingImag,
    };
    iterData.output = ComplexImagePair{
        .real = pongReal,
        .imag = pongImag,
    };
    iterData.ns *= iterData.r;
    // TODO: iterData.r=4;

    while (iterData.ns < outputDim)
    {
        doIteration(scopeAlloc.child_scope(), cb, iterData, nextFrame);
        const ComplexImagePair tmp = iterData.input;
        iterData.input = iterData.output;
        iterData.output = tmp;
        iterData.ns *= iterData.r;
    }

    // Columns next
    iterData.ns = 1;
    iterData.r = 2;
    iterData.transpose = !inverse;
    // TODO: Start with  R=4 if divides evenly already?
    doIteration(scopeAlloc.child_scope(), cb, iterData, nextFrame);
    {
        const ComplexImagePair tmp = iterData.input;
        iterData.input = iterData.output;
        iterData.output = tmp;
    }
    iterData.ns *= iterData.r;
    // TODO: iterData.r=4;

    while (iterData.ns < outputDim)
    {
        doIteration(scopeAlloc.child_scope(), cb, iterData, nextFrame);
        const ComplexImagePair tmp = iterData.input;
        iterData.input = iterData.output;
        iterData.output = tmp;
        iterData.ns *= iterData.r;
    }

    gRenderResources.images->release(iterData.output.real);
    gRenderResources.images->release(iterData.output.imag);

    const ComplexImagePair ret{
        .real = inverse ? iterData.input.imag : iterData.input.real,
        .imag = inverse ? iterData.input.real : iterData.input.imag,
    };

    return ret;
}

void BloomFft::doIteration(
    wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    const IterationData &iterData, uint32_t nextFrame)
{
    const vk::Extent2D inputExtent = getExtent2D(iterData.input.real);
    const vk::Extent2D outputExtent = getExtent2D(iterData.output.real);

    m_computePass.updateDescriptorSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{{
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(iterData.input.real).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(iterData.input.imag).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(iterData.output.real)
                        .view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(iterData.output.imag)
                        .view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
        }});

    transition(
        WHEELS_MOV(scopeAlloc), cb,
        Transitions{
            .images = StaticArray<ImageTransition, 4>{{
                {iterData.input.real, ImageState::ComputeShaderRead},
                {iterData.input.imag, ImageState::ComputeShaderRead},
                {iterData.output.real, ImageState::ComputeShaderWrite},
                {iterData.output.imag, ImageState::ComputeShaderWrite},
            }},
        });

    vk::DescriptorSet const descriptorSet = m_computePass.storageSet(nextFrame);

    const FftPC pcBlock{
        .inputResolution =
            uvec2{
                inputExtent.width,
                inputExtent.height,
            },
        .outputResolution =
            uvec2{
                outputExtent.width,
                outputExtent.height,
            },
        .ns = iterData.ns,
        .r = iterData.r,
        .flags = iterData.transpose ? 1u : 0u,
    };
    const uvec3 groupCount = m_computePass.groupCount(uvec3{
        (iterData.transpose ? outputExtent.height : outputExtent.width) /
            iterData.r,
        iterData.transpose ? outputExtent.width : outputExtent.height,
        1,
    });
    m_computePass.record(cb, pcBlock, groupCount, Span{&descriptorSet, 1});
}
