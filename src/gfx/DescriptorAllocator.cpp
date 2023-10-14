#include "DescriptorAllocator.hpp"

#include "../utils/Utils.hpp"
#include "VkUtils.hpp"

using namespace wheels;

namespace
{

// On Turing, these defaults require ~2MB of host memory per pool
constexpr auto sDefaultAccelerationStructureCount = 100;
constexpr auto sDefaultSamplerDescriptorCount = 100;
constexpr auto sDefaultDescriptorCount = 1000;
constexpr auto sDefaultDescriptorSetCount = 1000;
constexpr std::array sDefaultPoolSizes{
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
};

} // namespace

DescriptorAllocator::DescriptorAllocator(
    Allocator &alloc, Device *device, vk::DescriptorPoolCreateFlags flags)
: _device{device}
, _pools{alloc}
, _flags{flags}
{
    WHEELS_ASSERT(_device != nullptr);

    nextPool();
}

DescriptorAllocator::~DescriptorAllocator()
{
    if (_device != nullptr)
        for (auto &p : _pools)
            _device->logical().destroy(p);
}

void DescriptorAllocator::resetPools()
{
    for (auto &p : _pools)
        _device->logical().resetDescriptorPool(p);
    _activePool = 0;
}

vk::DescriptorSet DescriptorAllocator::allocate(
    const vk::DescriptorSetLayout &layout)
{
    vk::DescriptorSet ret;
    allocate(Span{&layout, 1}, Span{&ret, 1}, nullptr);
    return ret;
}

vk::DescriptorSet DescriptorAllocator::allocate(
    const vk::DescriptorSetLayout &layout, uint32_t variableDescriptorCount)
{
    const vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCounts{
        .descriptorSetCount = 1,
        .pDescriptorCounts = &variableDescriptorCount,
    };
    vk::DescriptorSet ret;
    allocate(
        Span{&layout, 1}, Span{&ret, 1},
        reinterpret_cast<const void *>(&variableCounts));
    return ret;
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
    Span<const vk::DescriptorSetLayout> layouts, Span<vk::DescriptorSet> output)
{
    return allocate(layouts, output, nullptr);
}

void DescriptorAllocator::allocate(
    Span<const vk::DescriptorSetLayout> layouts, Span<vk::DescriptorSet> output,
    const void *allocatePNext)
{
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
