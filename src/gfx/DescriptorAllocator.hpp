#ifndef PROSPER_GFX_DESCRIPTOR_ALLOCATOR_HPP
#define PROSPER_GFX_DESCRIPTOR_ALLOCATOR_HPP

#include "Allocators.hpp"
#include "gfx/Fwd.hpp"

#include <vulkan/vulkan.hpp>
#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/span.hpp>

// Basic idea from
// https://vkguide.dev/docs/extra-chapter/abstracting_descriptors/

class DescriptorAllocator
{
  public:
    DescriptorAllocator() noexcept = default;
    // Descriptors allocated by this allocator are implicitly freed when the
    // pools are destroyed
    ~DescriptorAllocator();

    DescriptorAllocator(DescriptorAllocator const &) = delete;
    DescriptorAllocator(DescriptorAllocator &&) = delete;
    DescriptorAllocator &operator=(DescriptorAllocator const &) = delete;
    DescriptorAllocator &operator=(DescriptorAllocator &&) = delete;

    void init(
        vk::DescriptorPoolCreateFlags flags = vk::DescriptorPoolCreateFlags{});
    void destroy();

    // Reset frees all allocated descriptors/sets and makes the pools available
    // for new allocations
    void resetPools();

    // Free is not allowed for individual descriptors. resetPools() can be used
    // to free all descriptors allocated by this allocator.
    vk::DescriptorSet allocate(
        const vk::DescriptorSetLayout &layout, const char *debugName);
    vk::DescriptorSet allocate(
        const vk::DescriptorSetLayout &layout, const char *debugName,
        uint32_t variableDescriptorCount);
    void allocate(
        wheels::Span<const vk::DescriptorSetLayout> layouts,
        wheels::Span<const char *const> debugNames,
        wheels::Span<vk::DescriptorSet> output);

  private:
    void nextPool();
    void allocate(
        wheels::Span<const vk::DescriptorSetLayout> layouts,
        wheels::Span<const char *const> debugNames,
        wheels::Span<vk::DescriptorSet> output, const void *allocatePNext);

    // Any dynamic allocations need to be manually destroyed in destroy()
    bool m_initialized{false};
    int32_t m_activePool{-1};
    wheels::Array<vk::DescriptorPool> m_pools{gAllocators.general};
    vk::DescriptorPoolCreateFlags m_flags;
};

// This allocator should only be used for the descriptors that can live
// until the end of the program. As such, reset() shouldn't be called so
// that users can rely on the descriptors being there once allocated.
// This is depended on by Device and init()/destroy() order relative to other
// similar globals is handled in main()
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern DescriptorAllocator gStaticDescriptorsAlloc;

#endif // PROSPER_GFX_DESCRIPTOR_ALLOCATOR_HPP
