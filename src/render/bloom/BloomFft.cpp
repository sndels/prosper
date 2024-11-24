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
    bool needsRadix2{false};
};

uint32_t pcFlags(FftFlags flags)
{
    uint32_t ret = 0;

    ret |= (uint32_t)flags.transpose;
    ret |= (uint32_t)flags.inverse << 1;
    ret |= (uint32_t)flags.needsRadix2 << 2;

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
            // Three times that for inverse transform and forward pass on kernel
            // Twice that for inverse transform
            .perFrameRecordLimit = 7 * 2 * 2 * 3,
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

void BloomFft::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, ImageHandle &inputOutput,
    const uint32_t nextFrame, bool inverse, const char *debugPrefix)
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

    const vk::Extent2D fftExtent = getExtent2D(inputOutput);
    WHEELS_ASSERT(fftExtent.width == fftExtent.height);

    const uint32_t outputDim = fftExtent.width;
    WHEELS_ASSERT(outputDim >= sMinResolution);
    WHEELS_ASSERT(std::popcount(outputDim) == 1);
    // Shader has no bounds checks as it assumes no idle threads
    WHEELS_ASSERT(outputDim % sGroupSize * 4 == 0);

    String debugName{scopeAlloc};
    debugName.extend(debugPrefix);
    if (inverse)
        debugName.extend("Inv");
    debugName.extend("FftPong");

    const ImageHandle pongImage = gRenderResources.images->create(
        ImageDescription{
            .format = sFftFormat,
            .width = fftExtent.width,
            .height = fftExtent.height,
            .usageFlags = vk::ImageUsageFlagBits::eSampled |
                          vk::ImageUsageFlagBits::eStorage,
        },
        debugName.c_str());

    // Rows first
    DispatchData dispatchData{
        // For a real input image, this will consider rg/ba as complex pairs
        // to perform four transforms for the price of two. However, this has
        // implications when the DFT is used for convolution.
        // TODO: What are those implications
        .images = StaticArray{{
            inputOutput,
            pongImage,
        }},
        .n = outputDim,
        .transpose = false,
        .inverse = inverse,
        .needsRadix2 = !isPowerOf(outputDim, 4u),
    };
    dispatch(scopeAlloc.child_scope(), cb, dispatchData, nextFrame);
    bool flipInOut = dispatchData.needsRadix2;
    uint32_t ns = dispatchData.needsRadix2 ? 2 : 1;
    while (ns < outputDim)
    {
        ns *= 4;
        flipInOut = !flipInOut;
    }
    if (flipInOut)
        dispatchData.images = StaticArray{{
            dispatchData.images[1],
            dispatchData.images[0],
        }};

    dispatchData.transpose = true;
    dispatch(scopeAlloc.child_scope(), cb, dispatchData, nextFrame);
    flipInOut = dispatchData.needsRadix2;
    ns = dispatchData.needsRadix2 ? 2 : 1;
    while (ns < outputDim)
    {
        ns *= 4;
        flipInOut = !flipInOut;
    }
    if (flipInOut)
        dispatchData.images = StaticArray{{
            dispatchData.images[1],
            dispatchData.images[0],
        }};

    gRenderResources.images->release(dispatchData.images[1]);

    inputOutput = dispatchData.images[0];
}

void BloomFft::dispatch(
    wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    const DispatchData &dispatchData, uint32_t nextFrame)
{
    const uint32_t outputDim = dispatchData.n;
    WHEELS_ASSERT(
        outputDim % sGroupSize == 0 &&
        "FFT shader assumes the input is divisible by group size");

    m_computePass.updateDescriptorSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{
            DescriptorInfo{StaticArray{{
                vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(dispatchData.images[0])
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                },
                vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(dispatchData.images[1])
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                },
            }}},
        });

    transition(
        WHEELS_MOV(scopeAlloc), cb,
        Transitions{
            .images = StaticArray<ImageTransition, 2>{{
                {dispatchData.images[0], ImageState::ComputeShaderReadWrite},
                {dispatchData.images[1], ImageState::ComputeShaderReadWrite},
            }},
        });

    const vk::DescriptorSet descriptorSet = m_computePass.storageSet(nextFrame);

    const FftPC pcBlock{
        .n = outputDim,
        .flags = pcFlags(FftFlags{
            .transpose = dispatchData.transpose,
            .inverse = dispatchData.inverse,
            .needsRadix2 = dispatchData.needsRadix2,
        }),
    };
    const uvec3 groupCount = m_computePass.groupCount(uvec3{
        1,
        outputDim,
        1,
    });
    m_computePass.record(cb, pcBlock, groupCount, Span{&descriptorSet, 1});
}
