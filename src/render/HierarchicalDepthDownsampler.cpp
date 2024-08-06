#include "HierarchicalDepthDownsampler.hpp"

#include "gfx/VkUtils.hpp"
#include "render/ComputePass.hpp"
#include "render/RenderResources.hpp"
#include "utils/Profiler.hpp"
#include "utils/Utils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

const uint32_t sGroupSizeX = 256u;
const uint32_t sMaxMips = 12;
const vk::Format sHierarchicalDepthFormat = vk::Format::eR32Sfloat;

// Ported from ffx_spd.h, removed mip and offset calculations
void SpdSetup(
    uvec2 &dispatchThreadGroupCountXY, uint32_t &numWorkGroups,
    const uvec4 &rectInfo)
{
    const uint32_t endIndexX = (rectInfo[0] + rectInfo[2] - 1) /
                               64; // rectInfo[0] = left, rectInfo[2] = width
    const uint32_t endIndexY = (rectInfo[1] + rectInfo[3] - 1) /
                               64; // rectInfo[1] = top, rectInfo[3] = height

    dispatchThreadGroupCountXY[0] = endIndexX + 1;
    dispatchThreadGroupCountXY[1] = endIndexY + 1;

    numWorkGroups =
        (dispatchThreadGroupCountXY[0]) * (dispatchThreadGroupCountXY[1]);
}

struct PCBlock
{
    ivec2 topMipResolution;
    uint32_t numWorkGroupsPerSlice;
    uint32_t mips;
};

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/hiz_downsampler.comp",
        .debugName = String{alloc, "HierarchicalDepthDownsamplerCS"},
        .groupSize = uvec3{sGroupSizeX, 1u, 1u},
    };
}

} // namespace

HierarchicalDepthDownsampler::~HierarchicalDepthDownsampler()
{
    // Don't check for m_initialized as we might be cleaning up after a failed
    // init.
    gDevice.destroy(m_atomicCounter);
}

void HierarchicalDepthDownsampler::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(
        WHEELS_MOV(scopeAlloc), shaderDefinitionCallback,
        ComputePassOptions{
            // GBuffer HiZ before and after second culling pass
            .perFrameRecordLimit = 2,
        });
    // Don't use a shared resource as this is tiny and the clear can be skipped
    // after the first frame if we know nothing else uses it.
    m_atomicCounter = gDevice.createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = sizeof(uint32_t),
                .usage = vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::eStorageBuffer,
                .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            },
        .debugName = "HizDownsamplerCounter"});

    m_initialized = true;
}

void HierarchicalDepthDownsampler::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}
void HierarchicalDepthDownsampler::startFrame() { m_computePass.startFrame(); }

ImageHandle HierarchicalDepthDownsampler::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    ImageHandle inNonLinearDepth, const uint32_t nextFrame, StrSpan debugPrefix)
{
    WHEELS_ASSERT(m_initialized);

    String passName{scopeAlloc};
    passName.extend(debugPrefix);
    passName.extend("HiZDownsampler");

    PROFILER_CPU_SCOPE(passName.c_str());

    const Image &inDepth = gRenderResources.images->resource(inNonLinearDepth);
    WHEELS_ASSERT(
        inDepth.format == vk::Format::eD32Sfloat &&
        "Input depth precision doesn't match HiZ format");
    WHEELS_ASSERT(inDepth.extent.depth == 1);
    // 1 px wide/tall inputs won't behave well, but also probably won't happen
    WHEELS_ASSERT(inDepth.extent.width > 1);
    WHEELS_ASSERT(inDepth.extent.height > 1);

    // Only floor as we want the number of mips to generate, not including mip 0
    const uint32_t hizMipCount =
        asserted_cast<uint32_t>(floor(std::log2((static_cast<float>(
            std::max(inDepth.extent.width, inDepth.extent.height))))));
    // This should work up to 4k
    WHEELS_ASSERT(hizMipCount <= sMaxMips);
    const uint32_t hizMip0Width = inDepth.extent.width >> 1;
    const uint32_t hizMip0Height = inDepth.extent.height >> 1;

    uvec2 dispatchThreadGroupCountXY{};
    PCBlock pcBlock{
        .topMipResolution =
            ivec2{
                asserted_cast<int32_t>(inDepth.extent.width),
                asserted_cast<int32_t>(inDepth.extent.height),
            },
        .mips = hizMipCount,
    };
    const uvec4 rectInfo{0, 0, inDepth.extent.width, inDepth.extent.height};
    SpdSetup(
        dispatchThreadGroupCountXY, pcBlock.numWorkGroupsPerSlice, rectInfo);

    String outName{scopeAlloc};
    outName.extend(debugPrefix);
    outName.extend("HierarchicalDepth");

    const ImageHandle outHierarchicalDepth = gRenderResources.images->create(
        ImageDescription{
            .format = sHierarchicalDepthFormat,
            .width = hizMip0Width,
            .height = hizMip0Height,
            .mipCount = hizMipCount,
            .usageFlags = vk::ImageUsageFlagBits::eSampled |
                          vk::ImageUsageFlagBits::eStorage,
        },
        outName.c_str());

    const Span<const vk::ImageView> mipViews =
        gRenderResources.images->subresourceViews(outHierarchicalDepth);

    StaticArray<vk::DescriptorImageInfo, sMaxMips> outputInfos;
    {
        size_t i = 0;
        for (; i < hizMipCount; ++i)
            outputInfos[i] = vk::DescriptorImageInfo{
                .imageView = mipViews[i],
                .imageLayout = vk::ImageLayout::eGeneral,
            };
        // Fill the remaining descriptors with copies of the first one so we
        // won't have unbound descriptors. We could use VK_EXT_robustness2 and
        // null descriptors, but this seems like less of a hassle since we
        // shouldn't be accessing them anyway.
        for (; i < sMaxMips; ++i)
            outputInfos[i] = vk::DescriptorImageInfo{
                .imageView = mipViews[0],
                .imageLayout = vk::ImageLayout::eGeneral,
            };
    }

    m_computePass.updateDescriptorSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{{
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = inDepth.view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .sampler = gRenderResources.nearestSampler,
            }},
            DescriptorInfo{outputInfos},
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = m_atomicCounter.handle,
                .range = VK_WHOLE_SIZE,
            }},
        }});

    transition(
        WHEELS_MOV(scopeAlloc), cb,
        Transitions{
            .images = StaticArray<ImageTransition, 2>{{
                {inNonLinearDepth, ImageState::ComputeShaderSampledRead},
                {outHierarchicalDepth, ImageState::ComputeShaderReadWrite},
            }},
        });

    if (m_counterNotCleared)
    {
        m_atomicCounter.transition(cb, BufferState::TransferDst);
        // Only need to clear once as SPD will leave this zeroed when the
        // dispatch exits
        cb.fillBuffer(m_atomicCounter.handle, 0, m_atomicCounter.byteSize, 0);
        m_atomicCounter.transition(cb, BufferState::ComputeShaderReadWrite);
        m_counterNotCleared = false;
    }

    PROFILER_GPU_SCOPE(cb, passName.c_str());

    const vk::DescriptorSet descriptorSet = m_computePass.storageSet(nextFrame);

    const uvec3 groupCount =
        uvec3{dispatchThreadGroupCountXY[0], dispatchThreadGroupCountXY[1], 1u};
    m_computePass.record(cb, pcBlock, groupCount, Span{&descriptorSet, 1});

    return outHierarchicalDepth;
}
