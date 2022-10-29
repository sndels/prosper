#include "DescriptorAllocator.hpp"

#include "Utils.hpp"

namespace
{

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

constexpr vk::DescriptorPoolCreateInfo sDefaultPoolInfo{
    .maxSets = sDefaultDescriptorSetCount,
    .poolSizeCount = asserted_cast<uint32_t>(sDefaultPoolSizes.size()),
    .pPoolSizes = sDefaultPoolSizes.data(),
};

} // namespace

DescriptorAllocator::DescriptorAllocator(Device *device)
: _device{device}
{
    assert(_device != nullptr);

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

void DescriptorAllocator::nextPool()
{
    // initially -1 so this makes it 0 and allocates the first pool
    _activePool++;
    if (asserted_cast<size_t>(_activePool) >= _pools.size())
        _pools.push_back(
            _device->logical().createDescriptorPool(sDefaultPoolInfo));
}
