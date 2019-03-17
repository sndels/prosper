#include "Swapchain.hpp"

#include <algorithm>
#include <stdexcept>

#include "Constants.hpp"

namespace {
    vk::SurfaceFormatKHR selectSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
    {
        // We're free to take our pick (sRGB output with "regular" 8bit rgba buffer)
        if (availableFormats.size() == 1 &&
            availableFormats[0].format == vk::Format::eUndefined)
            return {vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};

        // Check if preferred sRGB format is present
        for (const auto& format : availableFormats) {
            if (format.format == vk::Format::eB8G8R8A8Unorm &&
                format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
                return format;
        }

        // Default to the first one if preferred was not present
        // Picking "best one" is also an option here
        return availableFormats[0];
    }

    vk::PresentModeKHR selectSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
    {
        // Default to fifo (double buffering)
        vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;

        for (const auto& mode : availablePresentModes) {
            // We'd like mailbox to implement triple buffering
            if (mode == vk::PresentModeKHR::eMailbox)
                return mode;
            // fifo is not properly supported by some drivers so use immediate if available
            else if (mode == vk::PresentModeKHR::eImmediate)
                bestMode = mode;
        }

        return bestMode;
    }

    vk::Extent2D selectSwapExtent(const vk::Extent2D& extent, const vk::SurfaceCapabilitiesKHR& capabilities)
    {
        // Check if we have a fixed extent
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
            return capabilities.currentExtent;

        // Pick best resolution from given bounds
        vk::Extent2D actualExtent = {};
        actualExtent.width = std::clamp(
            extent.width,
            capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width
        );
        actualExtent.height = std::clamp(
            extent.height,
            capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height
        );

        return actualExtent;
    }
}

SwapchainSupport querySwapchainSupport(vk::PhysicalDevice device, vk::SurfaceKHR surface)
{
    SwapchainSupport details;
    details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
    details.formats = device.getSurfaceFormatsKHR(surface);
    details.presentModes = device.getSurfacePresentModesKHR(surface);

    return details;
}

SwapchainConfig selectSwapchainConfig(Device* device, const vk::Extent2D& extent)
{
    SwapchainSupport swapchainSupport = querySwapchainSupport(
        device->physical(),
        device->surface()
    );

    SwapchainConfig config;
    config.transform = swapchainSupport.capabilities.currentTransform;
    // Prefer one extra image to limit waiting on internal operations
    config.imageCount = swapchainSupport.capabilities.minImageCount + 1;
    if (swapchainSupport.capabilities.maxImageCount > 0 &&
        config.imageCount > swapchainSupport.capabilities.maxImageCount)
        config.imageCount = swapchainSupport.capabilities.maxImageCount;

    config.surfaceFormat = selectSwapSurfaceFormat(swapchainSupport.formats);
    config.presentMode = selectSwapPresentMode(swapchainSupport.presentModes);
    config.extent = selectSwapExtent(extent, swapchainSupport.capabilities);
    return config;
}

Swapchain::~Swapchain()
{
    destroy();
}

void Swapchain::create(Device* device, vk::RenderPass renderPass, const SwapchainConfig& config)
{
    _device = device;
    _config = config;
    createSwapchain();
    createImageViews();
    createFramebuffers(renderPass);
    createFences();
}

vk::Format Swapchain::format() const
{
    return _config.surfaceFormat.format;
}

const vk::Extent2D& Swapchain::extent() const
{
    return _config.extent;
}

uint32_t Swapchain::imageCount() const
{
    return _config.imageCount;
}

vk::Framebuffer Swapchain::fbo(size_t i)
{
    return i < _fbos.size() ? _fbos[i] : vk::Framebuffer();
}

size_t Swapchain::currentFrame() const
{
    return _currentFrame;
}

vk::Fence Swapchain::currentFence()
{
    return _inFlightFences[_currentFrame];
}

std::optional<uint32_t> Swapchain::acquireNextImage(const vk::Semaphore waitSemaphore)
{
    // Wait for last frame on fence to finish
    _device->logical().waitForFences(
        1, // fenceCount
        &_inFlightFences[_currentFrame],
        VK_TRUE, // waitAll
        std::numeric_limits<uint64_t>::max() // timeout
    );
    _device->logical().resetFences(
        1, // fenceCount
        &_inFlightFences[_currentFrame])
    ;

    // Get index of the next swap image
    const vk::Result result = _device->logical().acquireNextImageKHR(
        _swapchain,
        std::numeric_limits<uint64_t>::max(), // timeout
        waitSemaphore,
        {}, // fence
        &_nextImage
    );

    // Signal to recreate swap chain if out of date or suboptimal
    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR)
        return std::nullopt;
    else if (result != vk::Result::eSuccess)
        throw std::runtime_error("Failed to acquire swapchain image");

    return _nextImage;
}

bool Swapchain::present(uint32_t waitSemaphoreCount, const vk::Semaphore* waitSemaphores)
{
    const vk::PresentInfoKHR presentInfo(
        waitSemaphoreCount,
        waitSemaphores,
        1, // swapchainCount
        &_swapchain,
        &_nextImage
    );
    vk::Result result = _device->presentQueue().presentKHR(&presentInfo);

    // Signal to recreate swap chain if out of date or suboptimal
    bool good_swap = true;
    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR)
        good_swap = false;
    else if (result != vk::Result::eSuccess)
        throw std::runtime_error("Failed to present swapchain image");

    _currentFrame = (_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    return good_swap;
}

void Swapchain::createSwapchain()
{
    QueueFamilies indices = _device->queueFamilies();
    const std::array<uint32_t, 2> queueFamilyIndices = {{
        indices.graphicsFamily.value(),
        indices.presentFamily.value()
    }};

    // Conditional info
    // Handle ownership of images
    vk::SharingMode imageSharingMode;
    uint32_t queueFamilyIndexCount;
    const uint32_t* pQueueFamilyIndices;
    if (indices.graphicsFamily != indices.presentFamily) {
        // Pick concurrent to skip in-depth ownership jazz for now
        imageSharingMode = vk::SharingMode::eConcurrent;
        queueFamilyIndexCount = queueFamilyIndices.size();
        pQueueFamilyIndices = queueFamilyIndices.data();
    } else {
        imageSharingMode = vk::SharingMode::eExclusive;
        queueFamilyIndexCount = 0; // optional
        pQueueFamilyIndices = nullptr; // optional
    }

    // Create swapchain
    vk::SwapchainCreateInfoKHR createInfo(
        {}, // flags
        _device->surface(),
        _config.imageCount,
        _config.surfaceFormat.format,
        _config.surfaceFormat.colorSpace,
        _config.extent,
        1, // layers
        vk::ImageUsageFlagBits::eColorAttachment,
        imageSharingMode,
        queueFamilyIndexCount,
        pQueueFamilyIndices,
        _config.transform, // Can do mirrors, flips automagically
        vk::CompositeAlphaFlagBitsKHR::eOpaque,
        _config.presentMode,
        VK_TRUE // Don't care about pixels covered by other windows
    );

    _swapchain = _device->logical().createSwapchainKHR(createInfo);

    // Get swapchain images
    _images = _device->logical().getSwapchainImagesKHR(_swapchain);

}

void Swapchain::createImageViews()
{
    // Create simple image views to treat swap chain images as color targets
    for (auto& image : _images) {
        vk::ImageViewCreateInfo createInfo(
            {}, // flags
            image,
            vk::ImageViewType::e2D,
            _config.surfaceFormat.format,
            vk::ComponentMapping(), // Identity swizzles
            vk::ImageSubresourceRange(
                vk::ImageAspectFlagBits::eColor,
                0, // base mip
                1, // level count
                0, // base array layer
                1  // layer count
            )
        );
        _imageViews.push_back(_device->logical().createImageView(createInfo));
    }
}

void Swapchain::createFramebuffers(vk::RenderPass renderPass)
{
    // Create framebuffers for image views
    for (auto& view : _imageViews) {
        const std::array<vk::ImageView, 1> attachments = {{
            view
        }};

        vk::FramebufferCreateInfo framebufferInfo(
            {}, // flags
            renderPass,
            attachments.size(),
            attachments.data(),
            _config.extent.width,
            _config.extent.height,
            1 // layers
        );
        _fbos.push_back(_device->logical().createFramebuffer(framebufferInfo));
    }
}

void Swapchain::createFences()
{
    vk::FenceCreateInfo fenceInfo(
        vk::FenceCreateFlagBits::eSignaled
    );
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        _inFlightFences.push_back(_device->logical().createFence(fenceInfo));
}

void Swapchain::destroy()
{
    // Destroy vulkan resources
    for (auto fence : _inFlightFences)
        _device->logical().destroy(fence);
    for (auto framebuffer : _fbos)
        _device->logical().destroy(framebuffer);
    for (auto imageView : _imageViews)
        _device->logical().destroy(imageView);
    _device->logical().destroy(_swapchain);

    // Also clear the handles
    _swapchain = vk::SwapchainKHR();
    _images.clear();
    _imageViews.clear();
    _fbos.clear();
    _inFlightFences.clear();
    _currentFrame = 0;
    _nextImage = 0;
}
