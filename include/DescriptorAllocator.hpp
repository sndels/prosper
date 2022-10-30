#ifndef PROSPER_DESCRIPTOR_ALLOCATOR_HPP
#define PROSPER_DESCRIPTOR_ALLOCATOR_HPP

#include "Device.hpp"

#include <span>

// Basic idea from
// https://vkguide.dev/docs/extra-chapter/abstracting_descriptors/

class DescriptorAllocator
{
  public:
    DescriptorAllocator(Device *device);
    // Descriptors allocated by this allocator are implicitly freed when the
    // pools are destroyed
    ~DescriptorAllocator();

    DescriptorAllocator(DescriptorAllocator const &) = delete;
    DescriptorAllocator(DescriptorAllocator &&) = delete;
    DescriptorAllocator &operator=(DescriptorAllocator const &) = delete;
    DescriptorAllocator &operator=(DescriptorAllocator &&) = delete;

    // Reset frees all allocated descriptors/sets and makes the pools available
    // for new allocations
    void resetPools();

    // Free is not allowed for individual descriptors. resetPools() can be used
    // to free all descriptors allocated by this allocator.
    vk::DescriptorSet allocate(const vk::DescriptorSetLayout &layout);
    vk::DescriptorSet allocate(
        const vk::DescriptorSetLayout &layout,
        uint32_t variableDescriptorCount);
    std::vector<vk::DescriptorSet> allocate(
        std::span<const vk::DescriptorSetLayout> layouts);

  private:
    void nextPool();
    std::vector<vk::DescriptorSet> allocate(
        std::span<const vk::DescriptorSetLayout> layouts,
        const void *allocatePNext);

    Device *_device{nullptr};
    int32_t _activePool{-1};
    std::vector<vk::DescriptorPool> _pools;
};

#endif // PROSPER_DESCRIPTOR_ALLOCATOR_HPP
