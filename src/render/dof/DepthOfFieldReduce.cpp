#include "DepthOfFieldReduce.hpp"

#include <imgui.h>

#include "../../gfx/VkUtils.hpp"
#include "../../scene/Camera.hpp"
#include "../../utils/Profiler.hpp"
#include "../../utils/Utils.hpp"
#include "../RenderResources.hpp"
#include "../RenderTargets.hpp"

using namespace glm;
using namespace wheels;

namespace
{

const uint32_t sGroupSizeX = 256u;
const uint32_t sMaxMips = 12;

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
        .relPath = "shader/dof/reduce.comp",
        .debugName = String{alloc, "DepthOfFieldReduceCS"},
        .groupSize = uvec3{sGroupSizeX, 1u, 1u},
    };
}

} // namespace

DepthOfFieldReduce::~DepthOfFieldReduce()
{
    // Don't check for m_initialized as we might be cleaning up after a failed
    // init.
    gDevice.destroy(m_atomicCounter);
}

void DepthOfFieldReduce::init(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(
        WHEELS_MOV(scopeAlloc), staticDescriptorsAlloc,
        shaderDefinitionCallback);
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
        .debugName = "DofReduceCounter"});

    m_initialized = true;
}

void DepthOfFieldReduce::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

void DepthOfFieldReduce::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    const ImageHandle &inOutIlluminationMips, const uint32_t nextFrame,
    Profiler *profiler)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE(profiler, "  Reduce");

    const Image &inOutRes =
        gRenderResources.images->resource(inOutIlluminationMips);
    WHEELS_ASSERT(inOutRes.extent.depth == 1);
    // 0 mip is bound as source, rest as dst
    WHEELS_ASSERT(inOutRes.mipCount <= sMaxMips + 1);

    uvec2 dispatchThreadGroupCountXY{};
    PCBlock pcBlock{
        .topMipResolution =
            ivec2{
                asserted_cast<int32_t>(inOutRes.extent.width),
                asserted_cast<int32_t>(inOutRes.extent.height),
            },
        .mips = inOutRes.mipCount,
    };
    const uvec4 rectInfo{0, 0, inOutRes.extent.width, inOutRes.extent.height};
    SpdSetup(
        dispatchThreadGroupCountXY, pcBlock.numWorkGroupsPerSlice, rectInfo);

    // This is 1+mips for SPD as mip 0 is bound as the source and mip 1 is the
    // first destination
    const Span<const vk::ImageView> mipViews =
        gRenderResources.images->subresourceViews(inOutIlluminationMips);

    StaticArray<vk::DescriptorImageInfo, sMaxMips> outputInfos;
    {
        size_t i = 0;
        // Bind from view 1 onward as 0 is the source
        const size_t dstMipCount = mipViews.size() - 1;
        for (; i < dstMipCount; ++i)
            outputInfos[i] = vk::DescriptorImageInfo{
                .imageView = mipViews[i + 1],
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
                .imageView = mipViews[0],
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{outputInfos},
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = m_atomicCounter.handle,
                .range = VK_WHOLE_SIZE,
            }},
        }});

    gRenderResources.images->transition(
        cb, inOutIlluminationMips, ImageState::ComputeShaderReadWrite);

    if (m_counterNotCleared)
    {
        m_atomicCounter.transition(cb, BufferState::TransferDst);
        // Only need to clear once as SPD will leave this zeroed when the
        // dispatch exits
        cb.fillBuffer(m_atomicCounter.handle, 0, m_atomicCounter.byteSize, 0);
        m_atomicCounter.transition(cb, BufferState::ComputeShaderReadWrite);
        m_counterNotCleared = false;
    }

    PROFILER_GPU_SCOPE(profiler, cb, "  Reduce");

    const vk::DescriptorSet descriptorSet = m_computePass.storageSet(nextFrame);

    // Compute pass calculates group counts assuming extent / groupSize
    // TODO:
    // Rethink this interface. record() might be better off taking in group size
    // and there could be a groupSize(extent)-method that can be used to get the
    // typical calculation.
    const uvec3 extent = uvec3{
        dispatchThreadGroupCountXY[0] * sGroupSizeX,
        dispatchThreadGroupCountXY[1], 1u};
    m_computePass.record(cb, pcBlock, extent, Span{&descriptorSet, 1});
}
