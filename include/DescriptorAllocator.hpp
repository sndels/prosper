#ifndef PROSPER_DESCRIPTOR_ALLOCATOR_HPP
#define PROSPER_DESCRIPTOR_ALLOCATOR_HPP

#include "Device.hpp"
#include "VkUtils.hpp"

#include <span>

// Basic idea from
// https://vkguide.dev/docs/extra-chapter/abstracting_descriptors/

class DescriptorAllocator
{
  public:
    DescriptorAllocator(Device *device);
    ~DescriptorAllocator();

    DescriptorAllocator(DescriptorAllocator const &) = delete;
    DescriptorAllocator(DescriptorAllocator &&) = delete;
    DescriptorAllocator &operator=(DescriptorAllocator const &) = delete;
    DescriptorAllocator &operator=(DescriptorAllocator &&) = delete;

    void resetPools();

    vk::DescriptorSet allocate(const vk::DescriptorSetLayout &layout);
    vk::DescriptorSet allocate(
        const vk::DescriptorSetLayout &layout,
        uint32_t variableDescriptorCount);
    template <size_t N>
    std::vector<vk::DescriptorSet> allocate(
        std::span<const vk::DescriptorSetLayout, N> layouts);

  private:
    void nextPool();

    template <size_t N>
    std::vector<vk::DescriptorSet> allocate(
        std::span<const vk::DescriptorSetLayout, N> layouts,
        const void *allocatePNext);

    Device *_device{nullptr};
    int32_t _activePool{-1};
    std::vector<vk::DescriptorPool> _pools;
};

template <size_t N>
std::vector<vk::DescriptorSet> DescriptorAllocator::allocate(
    std::span<const vk::DescriptorSetLayout, N> layouts)
{
    return allocate(layouts, nullptr);
}

template <size_t N>
std::vector<vk::DescriptorSet> DescriptorAllocator::allocate(
    std::span<const vk::DescriptorSetLayout, N> layouts,
    const void *allocatePNext)
{
    std::vector<vk::DescriptorSet> ret;
    ret.resize(layouts.size());

    auto tryAllocate = [&]() -> vk::Result
    {
        const vk::DescriptorSetAllocateInfo info{
            .pNext = allocatePNext,
            .descriptorPool = _pools[_activePool],
            .descriptorSetCount = asserted_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data(),
        };
        return _device->logical().allocateDescriptorSets(&info, ret.data());
    };

    auto result = tryAllocate();
    // Get a new pool if we run out of the current one, just accept
    // failure if we run out of host or devie memory
    if (result == vk::Result::eErrorFragmentedPool ||
        result == vk::Result::eErrorOutOfPoolMemory)
    {
        nextPool();
        result = tryAllocate();
    }
    checkSuccess(result, "allocateDescriptorSets");

    return ret;
}

#endif // PROSPER_DESCRIPTOR_ALLOCATOR_HPP
