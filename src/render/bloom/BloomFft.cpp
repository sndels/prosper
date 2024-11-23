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

const uint32_t sGroupSize = 64;

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/bloom/fft.comp",
        .debugName = String{alloc, "BloomFftCS"},
        .groupSize = {sGroupSize, 1, 1},
    };
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
bool isPowerOf(uint32_t n, uint32_t base)
{
    uint32_t v = base;
    while (v < n)
        v *= base;

    return v == n;
}

struct FftFlags
{
    bool transpose{false};
    bool inverse{false};
};

uint32_t pcFlags(FftFlags flags)
{
    uint32_t ret = 0;

    ret |= (uint32_t)flags.transpose;
    ret |= (uint32_t)flags.inverse << 1;

    return ret;
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

ImageHandle BloomFft::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, ImageHandle input,
    const uint32_t nextFrame, bool inverse)
{
    WHEELS_ASSERT(m_initialized);

    // TODO:
    // - Twiddle LUT
    // - Shared memory version
    // - Does the two-for-one trick, input rg/ba as complex pairs, just work?
    //   - Seems reasonable that FFT-IFFT without any convolution/filtering just
    //     works, but seems like there should be some extra calculation when
    //     convolution is done. UE4 FFT Bloom stream mentioned inversion of the
    //     trick after IFFT, which sounds odd.
    // - Make sure convolution actually works as expected on the DFT signal
    //   - Transforming the input image as if rg/ba are complex pairs garbles
    //     the transform so need to recover it leveraging symmetries
    //     before convolution.
    // - Compare to DIT Cooley-Tukey
    //   - Ryg makes a convicing argument for that, also some FMA optimizations
    //     https://fgiesen.wordpress.com/2023/03/19/notes-on-ffts-for-implementers/

    PROFILER_CPU_GPU_SCOPE(cb, inverse ? "  InverseFft" : "  Fft");

    const vk::Extent2D inputExtent = getExtent2D(input);
    const uint32_t outputDim =
        std::bit_ceil(std::max(inputExtent.width, inputExtent.height));
    const vk::Extent2D outputExtent{
        .width = outputDim,
        .height = outputDim,
    };
    const ImageDescription targetDesc{
        .format = sFftFormat,
        .width = outputExtent.width,
        .height = outputExtent.height,
        .usageFlags =
            vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
    };
    const ImageHandle pingImage = gRenderResources.images->create(
        targetDesc, inverse ? "BloomInvFftPing" : "BloomFftPing");
    const ImageHandle pongImage = gRenderResources.images->create(
        targetDesc, inverse ? "BloomInvFftPong" : "BloomFftPong");

    const bool needsRadix2 = !isPowerOf(outputDim, 4u);
    // Rows first
    IterationData iterData{
        // For a real input image, this will consider rg/ba as complex pairs
        // to perform four transforms for the price of two. However, this has
        // implications when the DFT is used for convolution.
        // TODO: What are those implications
        .input = input,
        .output = pingImage,
        .ns = 1,
        .r = needsRadix2 ? 2u : 4u,
        .transpose = false,
        .inverse = inverse,
    };
    doIteration(scopeAlloc.child_scope(), cb, iterData, nextFrame);
    iterData.input = pingImage;
    iterData.output = pongImage;
    iterData.ns *= iterData.r;
    iterData.r = 4;

    while (iterData.ns < outputDim)
    {
        doIteration(scopeAlloc.child_scope(), cb, iterData, nextFrame);
        const ImageHandle tmp = iterData.input;
        iterData.input = iterData.output;
        iterData.output = tmp;
        iterData.ns *= iterData.r;
    }

    // Columns next
    iterData.ns = 1;
    iterData.r = needsRadix2 ? 2u : 4u;
    iterData.transpose = true;
    doIteration(scopeAlloc.child_scope(), cb, iterData, nextFrame);
    {
        const ImageHandle tmp = iterData.input;
        iterData.input = iterData.output;
        iterData.output = tmp;
    }
    iterData.ns *= iterData.r;
    iterData.r = 4;

    while (iterData.ns < outputDim)
    {
        doIteration(scopeAlloc.child_scope(), cb, iterData, nextFrame);
        const ImageHandle tmp = iterData.input;
        iterData.input = iterData.output;
        iterData.output = tmp;
        iterData.ns *= iterData.r;
    }

    gRenderResources.images->release(iterData.output);

    return iterData.input;
}

void BloomFft::doIteration(
    wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    const IterationData &iterData, uint32_t nextFrame)
{
    const vk::Extent2D inputExtent = getExtent2D(iterData.input);
    const vk::Extent2D outputExtent = getExtent2D(iterData.output);
    WHEELS_ASSERT(outputExtent.width == outputExtent.height);
    const uint32_t outputDim = outputExtent.width;
    WHEELS_ASSERT(
        outputDim % sGroupSize == 0 &&
        "FFT shader assumes the input is divisible by group size");

    m_computePass.updateDescriptorSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{{
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(iterData.input).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(iterData.output).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
        }});

    transition(
        WHEELS_MOV(scopeAlloc), cb,
        Transitions{
            .images = StaticArray<ImageTransition, 2>{{
                {iterData.input, ImageState::ComputeShaderRead},
                {iterData.output, ImageState::ComputeShaderWrite},
            }},
        });

    const vk::DescriptorSet descriptorSet = m_computePass.storageSet(nextFrame);

    const FftPC pcBlock{
        .inputResolution =
            uvec2{
                inputExtent.width,
                inputExtent.height,
            },
        .n = outputDim,
        .ns = iterData.ns,
        .r = iterData.r,
        .flags = pcFlags(FftFlags{
            .transpose = iterData.transpose,
            .inverse = iterData.inverse,
        }),
    };
    const uvec3 groupCount = m_computePass.groupCount(uvec3{
        outputDim / iterData.r,
        outputDim,
        1,
    });
    m_computePass.record(cb, pcBlock, groupCount, Span{&descriptorSet, 1});
}
