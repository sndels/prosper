#ifndef PROSPER_SWAPCHAIN_HPP
#define PROSPER_SWAPCHAIN_HPP

#include <vulkan/vulkan.hpp>

#include <optional>
#include <vector>

#include "Device.hpp"

struct SwapchainSupport
{
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;

    SwapchainSupport(vk::PhysicalDevice device, const vk::SurfaceKHR surface);
};

struct SwapchainConfig
{
    vk::SurfaceTransformFlagBitsKHR transform;
    vk::SurfaceFormatKHR surfaceFormat;
    vk::Format depthFormat;
    vk::PresentModeKHR presentMode;
    vk::Extent2D extent;
    uint32_t imageCount = 0;

    SwapchainConfig(
        std::shared_ptr<Device> device, const vk::Extent2D &preferredExtent);
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
    Swapchain(std::shared_ptr<Device> device, const SwapchainConfig &config);
    ~Swapchain();

    Swapchain(const Swapchain &other) = delete;
    Swapchain &operator=(const Swapchain &other) = delete;

    vk::Format format() const;
    const vk::Extent2D &extent() const;
    uint32_t imageCount() const;
    SwapchainImage image(size_t i) const;
    size_t nextFrame() const;
    vk::Fence currentFence() const;
    // nullopt tells to recreate swapchain
    std::optional<uint32_t> acquireNextImage(vk::Semaphore signalSemaphore);
    // false if swapchain should be recerated
    bool present(const std::array<vk::Semaphore, 1> &waitSemaphores);
    void recreate(const SwapchainConfig &config);

  private:
    void destroy();

    void createSwapchain();
    void createImages();
    void createFences();

    // Swapchain with null device is invalid or moved
    std::shared_ptr<Device> _device = nullptr;
    SwapchainConfig _config;

    vk::SwapchainKHR _swapchain;
    std::vector<SwapchainImage> _images;
    uint32_t _nextImage = 0;
    std::vector<vk::Fence> _inFlightFences;
    size_t _nextFrame = 0;
};

#endif // PROSPER_SWAPCHAIN_HPP
