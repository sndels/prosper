#ifndef PROSPER_DESCRIPTOR_ALLOCATOR_HPP
#define PROSPER_DESCRIPTOR_ALLOCATOR_HPP

#include "Device.hpp"

#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/span.hpp>

// Basic idea from
// https://vkguide.dev/docs/extra-chapter/abstracting_descriptors/

class DescriptorAllocator
{
  public:
    // Both alloc and device need to live as long as this
    DescriptorAllocator(
        wheels::Allocator &alloc, Device *device,
        vk::DescriptorPoolCreateFlags flags = vk::DescriptorPoolCreateFlags{});
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
    void allocate(
        wheels::Span<const vk::DescriptorSetLayout> layouts,
        wheels::Span<vk::DescriptorSet> output);

  private:
    void nextPool();
    void allocate(
        wheels::Span<const vk::DescriptorSetLayout> layouts,
        wheels::Span<vk::DescriptorSet> output, const void *allocatePNext);

    Device *_device{nullptr};
    int32_t _activePool{-1};
    wheels::Array<vk::DescriptorPool> _pools;
    vk::DescriptorPoolCreateFlags _flags;
};

#endif // PROSPER_DESCRIPTOR_ALLOCATOR_HPP
