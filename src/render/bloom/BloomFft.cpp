#include "BloomFft.hpp"

#include "render/RenderResources.hpp"
#include "render/Utils.hpp"
#include "utils/Logger.hpp"
#include "utils/Profiler.hpp"

#include <bit>
#include <imgui.h>
#include <shader_structs/push_constants/bloom/fft.h>

using namespace glm;
using namespace wheels;

namespace
{

const uint32_t sGroupSize = 32;

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/bloom/fft.comp",
        .debugName = String{alloc, "BloomFftCS"},
        .groupSize = {sGroupSize, 1, 1},
    };
}

uint32_t powerOfTwoIntoPower(uint32_t v)
{
    WHEELS_ASSERT(std::popcount(v) == 1);
    return 32 - std::countl_zero(v) - 1;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters) static
uint32_t firstRadixPower(uint32_t n, uint32_t maxRadix)
{
    uint32_t v = n;
    while (v > maxRadix)
        v /= maxRadix;

    return powerOfTwoIntoPower(v);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

struct FftConstants
{
    VkBool32 transpose{VK_FALSE};
    VkBool32 inverse{VK_FALSE};
    uint32_t radixPower{1};
};

uint32_t specializationIndex(FftConstants constants)
{
    uint32_t ret = 0;

    ret |= (uint32_t)constants.transpose;
    ret |= (uint32_t)constants.inverse << 1;
    // radixPower starts from 1, but we need tight indices
    ret |= (constants.radixPower - 1) << 2;

    return ret;
}

StaticArray<FftConstants, 16> generateSpecializationConstants()
{
    StaticArray<FftConstants, 16> ret;
    for (const VkBool32 transpose : {VK_FALSE, VK_TRUE})
    {
        for (const VkBool32 inverse : {VK_FALSE, VK_TRUE})
        {
            for (const uint32_t radixPower : {1, 2, 3, 4})
            {
                const FftConstants constants{
                    .transpose = transpose,
                    .inverse = inverse,
                    .radixPower = radixPower,
                };
                const uint32_t index = specializationIndex(constants);

                ret[index] = constants;
            }
        }
    }

    return ret;
}

} // namespace

void BloomFft::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    const StaticArray specializationConstants =
        generateSpecializationConstants();

    m_computePass.init(
        WHEELS_MOV(scopeAlloc), shaderDefinitionCallback,
        specializationConstants.span(),
        ComputePassOptions{
            // Single FFT run uses one set for first pass and two for the rest
            // for ping/pong binds.
            // We have at most three FFT runs per frame: kernel forward pass and
            // two passes for the convolution.
            .storageSetInstanceCount = 3 * 3,
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

    const vk::Extent2D fftExtent = getExtent2D(input);
    WHEELS_ASSERT(fftExtent.width == fftExtent.height);
    WHEELS_ASSERT(fftExtent.width >= sMinResolution);
    WHEELS_ASSERT(std::popcount(fftExtent.width) == 1);
    const uint32_t outputDim = fftExtent.width;
    WHEELS_ASSERT(outputDim % sGroupSize == 0);
    const uint32_t maxRadix = std::min(outputDim / sGroupSize, 16u);
    const uint32_t maxRadixPower = powerOfTwoIntoPower(maxRadix);

    String debugName{scopeAlloc};
    debugName.extend(debugPrefix);
    if (inverse)
        debugName.extend("Inv");
    debugName.extend("FftPing");

    const ImageDescription targetDesc{
        .format = sFftFormat,
        .width = fftExtent.width,
        .height = fftExtent.height,
        .usageFlags =
            vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
    };
    const ImageHandle pingImage =
        gRenderResources.images->create(targetDesc, debugName.c_str());
    debugName[debugName.size() - 3] = 'o';
    const ImageHandle pongImage =
        gRenderResources.images->create(targetDesc, debugName.c_str());

    const vk::DescriptorSet inputSet = m_computePass.updateStorageSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{{
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images->resource(input).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images->resource(pingImage).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
        }});
    const vk::DescriptorSet pingSet = m_computePass.updateStorageSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{{
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images->resource(pingImage).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images->resource(pongImage).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
        }});
    const vk::DescriptorSet pongSet = m_computePass.updateStorageSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{{
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images->resource(pongImage).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images->resource(pingImage).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
        }});

    // Rows first
    IterationData iterData{
        .descriptorSet = inputSet,
        // For a real input image, this will consider rg/ba as complex pairs
        // to perform four transforms for the price of two. However, this has
        // implications when the DFT is used for convolution.
        // TODO: What are those implications
        .input = input,
        .output = pingImage,
        .n = outputDim,
        .ns = 1,
        .radixPower = firstRadixPower(outputDim, maxRadix),
        .transpose = false,
        .inverse = inverse,
    };
    doIteration(scopeAlloc.child_scope(), cb, iterData);
    iterData.descriptorSet = pingSet;
    iterData.input = pingImage;
    iterData.output = pongImage;
    iterData.ns *= 1 << iterData.radixPower;
    iterData.radixPower = maxRadixPower;

    bool swapImages = false;
    while (iterData.ns < outputDim)
    {
        doIteration(scopeAlloc.child_scope(), cb, iterData);
        swapImages = !swapImages;
        iterData.descriptorSet = swapImages ? pongSet : pingSet;
        iterData.input = swapImages ? pongImage : pingImage;
        iterData.output = swapImages ? pingImage : pongImage;
        iterData.ns *= 1 << iterData.radixPower;
    }

    // Columns next
    iterData.ns = 1;
    iterData.radixPower = firstRadixPower(outputDim, maxRadix);
    iterData.transpose = true;
    doIteration(scopeAlloc.child_scope(), cb, iterData);
    swapImages = !swapImages;
    iterData.descriptorSet = swapImages ? pongSet : pingSet;
    iterData.input = swapImages ? pongImage : pingImage;
    iterData.output = swapImages ? pingImage : pongImage;
    iterData.ns *= 1 << iterData.radixPower;
    iterData.radixPower = maxRadixPower;

    while (iterData.ns < outputDim)
    {
        doIteration(scopeAlloc.child_scope(), cb, iterData);
        swapImages = !swapImages;
        iterData.descriptorSet = swapImages ? pongSet : pingSet;
        iterData.input = swapImages ? pongImage : pingImage;
        iterData.output = swapImages ? pingImage : pongImage;
        iterData.ns *= 1 << iterData.radixPower;
    }

    gRenderResources.images->release(iterData.output);

    return iterData.input;
}

void BloomFft::doIteration(
    wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    const IterationData &iterData)
{
    const uint32_t outputDim = iterData.n;
    WHEELS_ASSERT(
        outputDim / (1 << iterData.radixPower) % sGroupSize == 0 &&
        "FFT shader assumes the thread count is divisible by group size");

    transition(
        WHEELS_MOV(scopeAlloc), cb,
        Transitions{
            .images = StaticArray<ImageTransition, 2>{{
                {iterData.input, ImageState::ComputeShaderRead},
                {iterData.output, ImageState::ComputeShaderWrite},
            }},
        });

    const FftPC pcBlock{
        .n = outputDim,
        .ns = iterData.ns,
    };
    const uvec3 groupCount = m_computePass.groupCount(uvec3{
        iterData.radixPower == 4 ? (outputDim / (1 << iterData.radixPower)) * 4
                                 : outputDim / (1 << iterData.radixPower),
        outputDim,
        1,
    });
    m_computePass.record(
        cb, pcBlock, groupCount, Span{&iterData.descriptorSet, 1},
        ComputePassOptionalRecordArgs{
            .specializationIndex = specializationIndex(FftConstants{
                .transpose = iterData.transpose ? VK_TRUE : VK_FALSE,
                .inverse = iterData.inverse ? VK_TRUE : VK_FALSE,
                .radixPower = iterData.radixPower,
            }),
        });
}
