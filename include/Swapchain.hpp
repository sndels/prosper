#ifndef PROSPER_SWAPCHAIN_HPP
#define PROSPER_SWAPCHAIN_HPP

#include <vulkan/vulkan.hpp>

#include <wheels/allocators/allocator.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/span.hpp>
#include <wheels/containers/static_array.hpp>

#include "Device.hpp"
#include "Utils.hpp"

struct SwapchainSupport
{
    vk::SurfaceCapabilitiesKHR capabilities;
    wheels::Array<vk::SurfaceFormatKHR> formats;
    wheels::Array<vk::PresentModeKHR> presentModes;

    SwapchainSupport(
        wheels::Allocator &alloc, vk::PhysicalDevice device,
        vk::SurfaceKHR surface);
};

struct SwapchainConfig
{
    vk::SurfaceTransformFlagBitsKHR transform{
        vk::SurfaceTransformFlagBitsKHR::eIdentity};
    vk::SurfaceFormatKHR surfaceFormat;
    vk::Format depthFormat{vk::Format::eUndefined};
    vk::PresentModeKHR presentMode{vk::PresentModeKHR::eImmediate};
    vk::Extent2D extent;
    uint32_t imageCount{0};

    SwapchainConfig() = default;
    SwapchainConfig(
        wheels::ScopedScratch scopeAlloc, Device *device,
        const vk::Extent2D &preferredExtent);
};

struct SwapchainImage
{
    vk::Image handle;
    vk::Extent2D extent;
    vk::ImageSubresourceRange subresourceRange;
};

class Swapchain
{
  public:
    Swapchain(Device *device, const SwapchainConfig &config);
    ~Swapchain();

    Swapchain(const Swapchain &other) = delete;
    Swapchain(Swapchain &&other) = delete;
    Swapchain &operator=(const Swapchain &other) = delete;
    Swapchain &operator=(Swapchain &&other) = delete;

    [[nodiscard]] SwapchainConfig const &config() const;
    [[nodiscard]] vk::Format format() const;
    [[nodiscard]] const vk::Extent2D &extent() const;
    [[nodiscard]] uint32_t imageCount() const;
    [[nodiscard]] SwapchainImage image(size_t i) const;
    [[nodiscard]] size_t nextFrame() const;
    [[nodiscard]] vk::Fence currentFence() const;
    // nullopt tells to recreate swapchain
    [[nodiscard]] wheels::Optional<uint32_t> acquireNextImage(
        vk::Semaphore signalSemaphore);
    // false if swapchain should be recerated
    [[nodiscard]] bool present(
        wheels::Span<const vk::Semaphore> waitSemaphores);
    void recreate(const SwapchainConfig &config);

  private:
    void destroy();

    void createSwapchain();
    void createImages();
    void createFences();

    // Swapchain with null device is invalid or moved
    Device *_device{nullptr};
    SwapchainConfig _config;

    vk::SwapchainKHR _swapchain;
    wheels::StaticArray<SwapchainImage, MAX_SWAPCHAIN_IMAGES> _images;
    uint32_t _nextImage{0};
    wheels::StaticArray<vk::Fence, MAX_FRAMES_IN_FLIGHT> _inFlightFences;
    size_t _nextFrame{0};
};

#endif // PROSPER_SWAPCHAIN_HPP
