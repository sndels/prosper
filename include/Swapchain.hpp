#ifndef PROSPER_SWAPCHAIN_HPP
#define PROSPER_SWAPCHAIN_HPP

#include <vulkan/vulkan.hpp>

#include <optional>
#include <vector>

#include "Device.hpp"

struct SwapchainSupport {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};
SwapchainSupport querySwapchainSupport(vk::PhysicalDevice device, const vk::SurfaceKHR surface);

struct SwapchainConfig {
    vk::SurfaceTransformFlagBitsKHR transform;
    vk::SurfaceFormatKHR surfaceFormat;
    vk::Format depthFormat;
    vk::PresentModeKHR presentMode;
    vk::Extent2D extent;
    uint32_t imageCount;
};
SwapchainConfig selectSwapchainConfig(Device* device, const vk::Extent2D& extent);

struct SwapchainImage {
    vk::Image handle;
    vk::ImageView view;
    vk::Framebuffer fbo;
};

class Swapchain {
public:
    Swapchain() = default;
    ~Swapchain();

    Swapchain(const Swapchain& other) = delete;
    Swapchain& operator=(const Swapchain& other) = delete;

    void create(Device* device, const vk::RenderPass renderPass, const SwapchainConfig& config);
    void destroy();

    vk::Format format() const;
    const vk::Extent2D& extent() const;
    uint32_t imageCount() const;
    vk::Framebuffer fbo(size_t i) const;
    size_t nextFrame() const;
    vk::Fence currentFence() const;
    // nullopt tells to recreate swapchain
    std::optional<uint32_t> acquireNextImage(vk::Semaphore waitSemaphore);
    // false if swapchain should be recerated
    bool present(uint32_t waitSemaphoreCount, const vk::Semaphore* waitSemaphores);

private:
    void createSwapchain();
    void createImages();
    void createDepthResources();
    void createFramebuffers(const vk::RenderPass renderPass);
    void createFences();

    Device* _device = nullptr;
    SwapchainConfig _config = {};

    vk::SwapchainKHR _swapchain;
    std::vector<SwapchainImage> _images;
    uint32_t _nextImage = 0;
    Image _depthImage = {nullptr, nullptr, {}, nullptr};
    std::vector<vk::Fence> _inFlightFences;
    size_t _nextFrame = 0;
};

#endif // PROSPER_SWAPCHAIN_HPP
