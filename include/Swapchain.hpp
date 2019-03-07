#ifndef PROSPER_SWAPCHAIN_HPP
#define PROSPER_SWAPCHAIN_HPP

#include <vulkan/vulkan.h>

#include <optional>
#include <vector>

#include "Device.hpp"

struct SwapchainSupport {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};
SwapchainSupport querySwapchainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);

struct SwapchainConfig {
    VkSurfaceTransformFlagBitsKHR transform;
    VkSurfaceFormatKHR surfaceFormat;
    VkPresentModeKHR presentMode;
    VkExtent2D extent;
    uint32_t imageCount;
};
SwapchainConfig selectSwapchainConfig(Device* device, const VkExtent2D& extent);

class Swapchain {
public:
    Swapchain() = default;
    ~Swapchain();

    Swapchain(const Swapchain& other) = delete;
    Swapchain operator=(const Swapchain& other) = delete;

    void create(Device* device, VkRenderPass renderPass, const SwapchainConfig& config);
    void destroy();

    VkFormat format() const;
    const VkExtent2D& extent() const;
    size_t imageCount() const;
    VkFramebuffer fbo(size_t i);
    size_t currentFrame() const;
    VkFence currentFence();
    // nullopt tells to recreate swapchain
    std::optional<uint32_t> acquireNextImage(VkSemaphore waitSemaphore);
    // false if swapchain should be recerated
    bool present(uint32_t waitSemaphoreCount, VkSemaphore* waitSemaphores);

private:
    void createSwapchain();
    void createImageViews();
    void createFramebuffers(VkRenderPass renderPass);
    void createFences();

    Device* _device = nullptr;
    SwapchainConfig _config = {};

    VkSwapchainKHR _swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> _images;
    std::vector<VkImageView> _imageViews;
    std::vector<VkFramebuffer> _fbos;
    std::vector<VkFence> _inFlightFences;
    size_t _currentFrame = 0;
    uint32_t _nextImage = 0;
};

#endif // PROSPER_SWAPCHAIN_HPP
