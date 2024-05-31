#include "DescriptorAllocator.hpp"

#include "../utils/Utils.hpp"
#include "Device.hpp"
#include "VkUtils.hpp"

#include <wheels/containers/static_array.hpp>

using namespace wheels;

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

DescriptorAllocator::~DescriptorAllocator()
{
    if (_device != nullptr)
    {
        for (auto &p : _pools)
            _device->logical().destroy(p);
    }
}

void DescriptorAllocator::init(
    Device *device, vk::DescriptorPoolCreateFlags flags)
{
    WHEELS_ASSERT(!_initialized);
    WHEELS_ASSERT(device != nullptr);

    _device = device;
    _flags = flags;

    nextPool();

    _initialized = true;
}

void DescriptorAllocator::resetPools()
{
    WHEELS_ASSERT(_initialized);

    for (auto &p : _pools)
        _device->logical().resetDescriptorPool(p);
    _activePool = 0;
}

vk::DescriptorSet DescriptorAllocator::allocate(
    const vk::DescriptorSetLayout &layout)
{
    WHEELS_ASSERT(_initialized);

    vk::DescriptorSet ret;
    allocate(Span{&layout, 1}, Span{&ret, 1}, nullptr);
    return ret;
}

vk::DescriptorSet DescriptorAllocator::allocate(
    const vk::DescriptorSetLayout &layout, uint32_t variableDescriptorCount)
{
    WHEELS_ASSERT(_initialized);

    const vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCounts{
        .descriptorSetCount = 1,
        .pDescriptorCounts = &variableDescriptorCount,
    };
    vk::DescriptorSet ret;
    allocate(Span{&layout, 1}, Span{&ret, 1}, &variableCounts);
    return ret;
}

void DescriptorAllocator::allocate(
    Span<const vk::DescriptorSetLayout> layouts, Span<vk::DescriptorSet> output)
{
    WHEELS_ASSERT(_initialized);

    return allocate(layouts, output, nullptr);
}

void DescriptorAllocator::nextPool()
{
    // initially -1 so this makes it 0 and allocates the first pool
    _activePool++;
    if (asserted_cast<size_t>(_activePool) >= _pools.size())
        _pools.push_back(_device->logical().createDescriptorPool(
            vk::DescriptorPoolCreateInfo{
                .flags = _flags,
                .maxSets = sDefaultDescriptorSetCount,
                .poolSizeCount =
                    asserted_cast<uint32_t>(sDefaultPoolSizes.size()),
                .pPoolSizes = sDefaultPoolSizes.data(),
            }));
}

void DescriptorAllocator::allocate(
    Span<const vk::DescriptorSetLayout> layouts, Span<vk::DescriptorSet> output,
    const void *allocatePNext)
{
    WHEELS_ASSERT(_initialized);
    WHEELS_ASSERT(layouts.size() == output.size());

    auto tryAllocate = [&]() -> vk::Result
    {
        const vk::DescriptorSetAllocateInfo info{
            .pNext = allocatePNext,
            .descriptorPool = _pools[_activePool],
            .descriptorSetCount = asserted_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data(),
        };
        return _device->logical().allocateDescriptorSets(&info, output.data());
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
}
