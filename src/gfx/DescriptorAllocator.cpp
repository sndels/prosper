#include "DescriptorAllocator.hpp"

#include "gfx/Device.hpp"
#include "gfx/VkUtils.hpp"
#include "utils/Utils.hpp"

#include <wheels/containers/static_array.hpp>

using namespace wheels;

namespace gfx
{

namespace
{

// On Turing, these defaults require ~2MB of host memory per pool
constexpr auto sDefaultAccelerationStructureCount = 100;
constexpr auto sDefaultSamplerDescriptorCount = 100;
constexpr auto sDefaultDescriptorCount = 1000;
constexpr auto sDefaultDescriptorSetCount = 1000;
constexpr StaticArray sDefaultPoolSizes{{
    vk::DescriptorPoolSize{
        .type = vk::DescriptorType::eSampler,
        .descriptorCount = sDefaultSamplerDescriptorCount,
    },
    vk::DescriptorPoolSize{
        .type = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = sDefaultDescriptorCount,
    },
    vk::DescriptorPoolSize{
        .type = vk::DescriptorType::eSampledImage,
        .descriptorCount = sDefaultDescriptorCount,
    },
    vk::DescriptorPoolSize{
        .type = vk::DescriptorType::eStorageImage,
        .descriptorCount = sDefaultDescriptorCount,
    },
    vk::DescriptorPoolSize{
        .type = vk::DescriptorType::eUniformTexelBuffer,
        .descriptorCount = sDefaultDescriptorCount,
    },
    vk::DescriptorPoolSize{
        .type = vk::DescriptorType::eStorageTexelBuffer,
        .descriptorCount = sDefaultDescriptorCount,
    },
    vk::DescriptorPoolSize{
        .type = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = sDefaultDescriptorCount,
    },
    vk::DescriptorPoolSize{
        .type = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = sDefaultDescriptorCount,
    },
    vk::DescriptorPoolSize{
        .type = vk::DescriptorType::eUniformBufferDynamic,
        .descriptorCount = sDefaultDescriptorCount,
    },
    vk::DescriptorPoolSize{
        .type = vk::DescriptorType::eStorageBufferDynamic,
        .descriptorCount = sDefaultDescriptorCount,
    },
    vk::DescriptorPoolSize{
        .type = vk::DescriptorType::eInputAttachment,
        .descriptorCount = sDefaultDescriptorCount,
    },
    vk::DescriptorPoolSize{
        .type = vk::DescriptorType::eAccelerationStructureKHR,
        .descriptorCount = sDefaultAccelerationStructureCount,
    },
}};

} // namespace

// This used everywhere and init()/destroy() order relative to other similar
// globals is handled in main()
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DescriptorAllocator gStaticDescriptorsAlloc;

DescriptorAllocator::~DescriptorAllocator()
{
    WHEELS_ASSERT(!m_initialized && "destroy() not called?");
}

void DescriptorAllocator::init(vk::DescriptorPoolCreateFlags flags)
{
    WHEELS_ASSERT(!m_initialized);

    m_flags = flags;

    nextPool();

    m_initialized = true;
}

void DescriptorAllocator::destroy()
{
    // Don't check for m_initialized as we might be cleaning up after a failed
    // init.
    for (const vk::DescriptorPool p : m_pools)
        gDevice.logical().destroy(p);

    m_pools.~Array();

    m_initialized = false;
}

void DescriptorAllocator::resetPools()
{
    WHEELS_ASSERT(m_initialized);

    for (auto &p : m_pools)
        gDevice.logical().resetDescriptorPool(p);
    m_activePool = 0;
}

vk::DescriptorSet DescriptorAllocator::allocate(
    const vk::DescriptorSetLayout &layout, const char *debugName)
{
    WHEELS_ASSERT(m_initialized);

    vk::DescriptorSet ret;
    allocate(Span{&layout, 1}, Span{&debugName, 1}, Span{&ret, 1}, nullptr);
    return ret;
}

vk::DescriptorSet DescriptorAllocator::allocate(
    const vk::DescriptorSetLayout &layout, const char *debugName,
    uint32_t variableDescriptorCount)
{
    WHEELS_ASSERT(m_initialized);

    const vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCounts{
        .descriptorSetCount = 1,
        .pDescriptorCounts = &variableDescriptorCount,
    };
    vk::DescriptorSet ret;
    allocate(
        Span{&layout, 1}, Span{&debugName, 1}, Span{&ret, 1}, &variableCounts);
    return ret;
}

void DescriptorAllocator::allocate(
    Span<const vk::DescriptorSetLayout> layouts,
    Span<const char *const> debugNames, Span<vk::DescriptorSet> output)
{
    WHEELS_ASSERT(m_initialized);
    allocate(layouts, debugNames, output, nullptr);
}

void DescriptorAllocator::nextPool()
{
    // initially -1 so this makes it 0 and allocates the first pool
    m_activePool++;
    if (asserted_cast<size_t>(m_activePool) >= m_pools.size())
        m_pools.push_back(
            gDevice.logical().createDescriptorPool(vk::DescriptorPoolCreateInfo{
                .flags = m_flags,
                .maxSets = sDefaultDescriptorSetCount,
                .poolSizeCount =
                    asserted_cast<uint32_t>(sDefaultPoolSizes.size()),
                .pPoolSizes = sDefaultPoolSizes.data(),
            }));
}

void DescriptorAllocator::allocate(
    Span<const vk::DescriptorSetLayout> layouts,
    Span<const char *const> debugNames, Span<vk::DescriptorSet> output,
    const void *allocatePNext)
{
    WHEELS_ASSERT(m_initialized);
    WHEELS_ASSERT(layouts.size() == debugNames.size());
    WHEELS_ASSERT(layouts.size() == output.size());

    auto tryAllocate = [&]() -> vk::Result
    {
        const vk::DescriptorSetAllocateInfo info{
            .pNext = allocatePNext,
            .descriptorPool = m_pools[m_activePool],
            .descriptorSetCount = asserted_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data(),
        };
        return gDevice.logical().allocateDescriptorSets(&info, output.data());
    };

    auto result = tryAllocate();
    // Get a new pool if we run out of the current one, just accept
    // failure if we run out of host or device memory
    if (result == vk::Result::eErrorFragmentedPool ||
        result == vk::Result::eErrorOutOfPoolMemory)
    {
        nextPool();
        result = tryAllocate();
    }
    checkSuccess(result, "allocateDescriptorSets");
    if (result == vk::Result::eSuccess)
    {
        const size_t setsCount = layouts.size();
        for (size_t i = 0; i < setsCount; ++i)
        {
            gDevice.logical().setDebugUtilsObjectNameEXT(
                vk::DebugUtilsObjectNameInfoEXT{
                    .objectType = vk::ObjectType::eDescriptorSet,
                    .objectHandle = reinterpret_cast<uint64_t>(
                        static_cast<VkDescriptorSet>(output[i])),
                    .pObjectName = debugNames[i],
                });
        }
    }
}

} // namespace gfx
