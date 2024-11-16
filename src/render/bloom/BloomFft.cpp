#include "BloomFft.hpp"

#include "gfx/Device.hpp"
#include "render/RenderResources.hpp"
#include "render/Utils.hpp"
#include "utils/Profiler.hpp"

#include <bit>
#include <glm/gtc/constants.hpp>
#include <imgui.h>
#include <shader_structs/push_constants/bloom/fft.h>
#include <utility>

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

} // namespace

BloomFft::~BloomFft() { gDevice.destroy(mTwiddleLut); }

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
    // - Twiddle LUT
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
    const ImageDescription targetDesc{
        .format = sFftFormat,
        .width = outputExtent.width,
        .height = outputExtent.height,
        .usageFlags =
            vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
    };
    const ImageHandle pingReal = gRenderResources.images->create(
        targetDesc, inverse ? "BloomInvFftPingReal" : "BloomFftPingReal");
    const ImageHandle pingImag = gRenderResources.images->create(
        targetDesc, inverse ? "BloomInvFftPingImag" : "BloomFftPingImag");
    const ImageHandle pongReal = gRenderResources.images->create(
        targetDesc, inverse ? "BloomInvFftPongReal" : "BloomFftPongReal");
    const ImageHandle pongImag = gRenderResources.images->create(
        targetDesc, inverse ? "BloomInvFftPongImag" : "BloomFftPongImag");

    if (mTwiddleLutN != outputDim)
        generateTwiddleLut(scopeAlloc.child_scope(), outputDim);

    const bool needsRadix2 = !isPowerOf(outputDim, 4u);
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
        .r = needsRadix2 ? 2u : 4u,
        .transpose = false,
    };
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
    iterData.r = 4;

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
    iterData.r = needsRadix2 ? 2u : 4u;
    iterData.transpose = true;
    doIteration(scopeAlloc.child_scope(), cb, iterData, nextFrame);
    {
        const ComplexImagePair tmp = iterData.input;
        iterData.input = iterData.output;
        iterData.output = tmp;
    }
    iterData.ns *= iterData.r;
    iterData.r = 4;

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

void BloomFft::generateTwiddleLut(wheels::ScopedScratch scopeAlloc, uint32_t n)
{
    uint32_t totalTwiddleCount = 0;
    {
        uint32_t ns = n / 2;
        while (ns > 1)
        {
            totalTwiddleCount += ns;
            ns /= 2;
        }
    }
    totalTwiddleCount *= 4;

    Array<vec2> lut{scopeAlloc};
    lut.reserve(totalTwiddleCount);
    {
        // R=2 Ns=1 are trivial (1, 0) twiddles so let's skip them
        const uint32_t rCount = 4;
        uint32_t ns = 1;
        while (ns < n)
        {
            // TODO:
            // These have a lot of symmetries, only ns unique values to be found
            for (uint32_t j = 0; j < ns; ++j)
            {
                const float angle = -two_pi<float>() * static_cast<float>(j) /
                                    static_cast<float>(ns * rCount);
                lut.push_back(
                    vec2{std::cos(0.f * angle), std::sin(0.f * angle)});
                lut.push_back(
                    vec2{std::cos(1.f * angle), std::sin(1.f * angle)});
                lut.push_back(
                    vec2{std::cos(2.f * angle), std::sin(2.f * angle)});
                lut.push_back(
                    vec2{std::cos(3.f * angle), std::sin(3.f * angle)});
            }
            ns *= 2;
        }
    }

    gDevice.destroy(mTwiddleLut);
    mTwiddleLut = gDevice.createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = lut.size() * sizeof(float),
                .usage = vk::BufferUsageFlagBits::eStorageBuffer,
                .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            },
        .initialData = lut.data(),
        .debugName = "FftTwiddleLut",
    });
    mTwiddleLutN = n;
}

void BloomFft::doIteration(
    wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    const IterationData &iterData, uint32_t nextFrame)
{
    const vk::Extent2D inputExtent = getExtent2D(iterData.input.real);
    const vk::Extent2D outputExtent = getExtent2D(iterData.output.real);
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
                    gRenderResources.images->resource(iterData.input.real).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(iterData.input.imag).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = mTwiddleLut.handle,
                .range = VK_WHOLE_SIZE,
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
        .flags = iterData.transpose ? 1u : 0u,
    };
    const uvec3 groupCount = m_computePass.groupCount(uvec3{
        outputDim / iterData.r,
        outputDim,
        1,
    });
    m_computePass.record(cb, pcBlock, groupCount, Span{&descriptorSet, 1});
}
