#include "Swapchain.hpp"

#include <algorithm>
#include <stdexcept>

#include "Constants.hpp"

namespace {
    VkSurfaceFormatKHR selectSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
    {
        // We're free to take our pick (sRGB output with "regular" 8bit rgba buffer)
        if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED)
            return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};

        // Check if preferred sRGB format is present
        for (const auto& format : availableFormats) {
            if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
                format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                return format;
        }

        // Default to the first one if preferred was not present
        // Picking "best one" is also an option here
        return availableFormats[0];
    }

    VkPresentModeKHR selectSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
    {
        // Default to fifo (double buffering)
        VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

        for (const auto& mode : availablePresentModes) {
            // We'd like mailbox to implement triple buffering
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
                return mode;
            // fifo is not properly supported by some drivers so use immediate if available
            else if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
                bestMode = mode;
        }

        return bestMode;
    }

    VkExtent2D selectSwapExtent(const VkExtent2D& extent, const VkSurfaceCapabilitiesKHR& capabilities)
    {
        // Check if we have a fixed extent
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
            return capabilities.currentExtent;

        // Pick best resolution from given bounds
        VkExtent2D actualExtent = {};
        actualExtent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }
}

SwapchainSupport querySwapchainSupport(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    SwapchainSupport details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (formatCount > 0) {
        details.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    if (presentModeCount > 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

SwapchainConfig selectSwapchainConfig(Device* device, const VkExtent2D& extent)
{
    SwapchainSupport swapchainSupport = querySwapchainSupport(device->physicalDevice(), device->surface());

    SwapchainConfig config;
    config.transform = swapchainSupport.capabilities.currentTransform;
    // Prefer one extra image to limit waiting on internal operations
    config.imageCount = swapchainSupport.capabilities.minImageCount + 1;
    if (swapchainSupport.capabilities.maxImageCount > 0 && config.imageCount > swapchainSupport.capabilities.maxImageCount)
        config.imageCount = swapchainSupport.capabilities.maxImageCount;

    config.surfaceFormat = selectSwapSurfaceFormat(swapchainSupport.formats);
    config.presentMode = selectSwapPresentMode(swapchainSupport.presentModes);
    config.extent = selectSwapExtent(extent, swapchainSupport.capabilities);
    return config;
}

void Swapchain::cleanup()
{
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        vkDestroyFence(_device->handle(), _inFlightFences[i], nullptr);
    for (auto framebuffer : _fbos)
        vkDestroyFramebuffer(_device->handle(), framebuffer, nullptr);
    for (auto imageView : _imageViews)
        vkDestroyImageView(_device->handle(), imageView, nullptr);
    vkDestroySwapchainKHR(_device->handle(), _swapchain, nullptr);
}

void Swapchain::init(Device* device, VkRenderPass renderPass, const SwapchainConfig& config)
{
    _device = device;
    _config = config;
    createSwapchain();
    createImageViews();
    createFramebuffers(renderPass);
    createFences();
}

VkFormat Swapchain::format() const
{
    return _config.surfaceFormat.format;
}

const VkExtent2D& Swapchain::extent() const
{
    return _config.extent;
}

size_t Swapchain::imageCount() const
{
    return _config.imageCount;
}

VkFramebuffer Swapchain::fbo(size_t i)
{
    return i < _fbos.size() ? _fbos[i] : VK_NULL_HANDLE;
}

size_t Swapchain::currentFrame() const
{
    return _currentFrame;
}

VkFence Swapchain::currentFence()
{
    return _inFlightFences[_currentFrame];
}

uint32_t Swapchain::acquireNextImage(VkSemaphore waitSemaphore)
{
    // Wait for last frame on fence to finish
    vkWaitForFences(_device->handle(), 1, &_inFlightFences[_currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());
    vkResetFences(_device->handle(), 1, &_inFlightFences[_currentFrame]);

    // Get index of the next swap image
    vkAcquireNextImageKHR(_device->handle(), _swapchain, std::numeric_limits<uint64_t>::max(), waitSemaphore, VK_NULL_HANDLE, &_nextImage);

    return _nextImage;
}

void Swapchain::present(uint32_t waitSemaphoreCount, VkSemaphore* waitSemaphores)
{
    VkSwapchainKHR swapchains[] = {_swapchain};
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = waitSemaphoreCount;
    presentInfo.pWaitSemaphores = waitSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;
    presentInfo.pImageIndices = &_nextImage;
    presentInfo.pResults = nullptr; // optional

    vkQueuePresentKHR(_device->presentQueue(), &presentInfo);

    _currentFrame = (_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void Swapchain::createSwapchain()
{
    QueueFamilies indices = _device->queueFamilies();
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

    // Fill out info
    VkSwapchainCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = _device->surface();
    createInfo.minImageCount = _config.imageCount;
    createInfo.imageFormat = _config.surfaceFormat.format;
    createInfo.imageColorSpace = _config.surfaceFormat.colorSpace;
    createInfo.imageExtent = _config.extent;
    createInfo.imageArrayLayers = 1; // Always 1 if not stereoscopic
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    // Handle ownership of images
    if (indices.graphicsFamily != indices.presentFamily) {
        // Pick concurrent to skip in-depth ownership jazz for now
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0; // optional
        createInfo.pQueueFamilyIndices = nullptr; // optional
    }

    createInfo.preTransform = _config.transform; // Do mirrors, flips here
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // Opaque window
    createInfo.presentMode = _config.presentMode;
    createInfo.clipped = VK_TRUE; // Don't care about pixels covered by other windows
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(_device->handle(), &createInfo, nullptr, &_swapchain) != VK_SUCCESS)
        throw std::runtime_error("Failed to create swap chain");

    vkGetSwapchainImagesKHR(_device->handle(), _swapchain, &_config.imageCount, nullptr);
    _images.resize(_config.imageCount);
    vkGetSwapchainImagesKHR(_device->handle(), _swapchain, &_config.imageCount, _images.data());

}

void Swapchain::createImageViews()
{
    // Create simple image views to treat swap chain images as color targets
    _imageViews.resize(_config.imageCount);
    for (size_t i = 0; i < _imageViews.size(); ++i) {
        VkImageViewCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = _images[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = _config.surfaceFormat.format;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(_device->handle(), &createInfo, nullptr, &_imageViews[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create image views");
    }
}

void Swapchain::createFramebuffers(VkRenderPass renderPass)
{
    _fbos.resize(_config.imageCount);

    for (size_t i = 0; i < _fbos.size(); ++i) {
        VkImageView attachments[] = {
            _imageViews[i]
        };

        VkFramebufferCreateInfo  framebufferInfo = {};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = _config.extent.width;
        framebufferInfo.height = _config.extent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(_device->handle(), &framebufferInfo, nullptr, &_fbos[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create framebuffer");
    }
}

void Swapchain::createFences()
{
    _inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (vkCreateFence(_device->handle(), &fenceInfo, nullptr, &_inFlightFences[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create semaphores");
    }
}
