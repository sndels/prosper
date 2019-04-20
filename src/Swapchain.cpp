#include "Swapchain.hpp"

#include <algorithm>
#include <stdexcept>

#include "Constants.hpp"
#include "VkUtils.hpp"

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
        const vk::Extent2D actualExtent{
            std::clamp(
                extent.width,
                capabilities.minImageExtent.width,
                capabilities.maxImageExtent.width
            ),
            std::clamp(
                extent.height,
                capabilities.minImageExtent.height,
                capabilities.maxImageExtent.height
            )
        };

        return actualExtent;
    }
}

SwapchainSupport querySwapchainSupport(vk::PhysicalDevice device, const vk::SurfaceKHR surface)
{
    const SwapchainSupport details = {
        device.getSurfaceCapabilitiesKHR(surface),
        device.getSurfaceFormatsKHR(surface),
        device.getSurfacePresentModesKHR(surface)
    };

    return details;
}

SwapchainConfig selectSwapchainConfig(Device* device, const vk::Extent2D& extent)
{
    const SwapchainSupport swapchainSupport = querySwapchainSupport(
        device->physical(),
        device->surface()
    );

    SwapchainConfig config = {
        swapchainSupport.capabilities.currentTransform,
        selectSwapSurfaceFormat(swapchainSupport.formats),
        vk::Format::eD32Sfloat,
        selectSwapPresentMode(swapchainSupport.presentModes),
        selectSwapExtent(extent, swapchainSupport.capabilities),
        swapchainSupport.capabilities.minImageCount + 1 // Prefer one extra image to limit waiting on internal operations
    };
    if (swapchainSupport.capabilities.maxImageCount > 0 &&
        config.imageCount > swapchainSupport.capabilities.maxImageCount)
        config.imageCount = swapchainSupport.capabilities.maxImageCount;

    return config;
}

Swapchain::~Swapchain()
{
    destroy();
}

void Swapchain::create(Device* device, const vk::RenderPass renderPass, const SwapchainConfig& config)
{
    _device = device;
    _config = config;
    createSwapchain();
    createImages();
    createDepthResources();
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

vk::Framebuffer Swapchain::fbo(size_t i) const
{
    if (i < _images.size())
        return _images[i].fbo;
    throw std::runtime_error("Tried to index past swap image count");
}

size_t Swapchain::nextFrame() const
{
    return _nextFrame;
}

vk::Fence Swapchain::currentFence() const
{
    return _inFlightFences[_nextFrame];
}

std::optional<uint32_t> Swapchain::acquireNextImage(vk::Semaphore waitSemaphore)
{
    _device->logical().waitForFences(
        1, // fenceCount
        &_inFlightFences[_nextFrame],
        VK_TRUE, // waitAll
        std::numeric_limits<uint64_t>::max() // timeout
    );
    _device->logical().resetFences(
        1, // fenceCount
        &_inFlightFences[_nextFrame])
    ;

    // TODO: noexcept, modern interface would throw on ErrorOutOfDate
    const auto result = _device->logical().acquireNextImageKHR(
        _swapchain,
        std::numeric_limits<uint64_t>::max(), // timeout
        waitSemaphore,
        {}, // fence
        &_nextImage
    );

    // Swapchain should be recreated if out of date or suboptimal
    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR)
        return std::nullopt;
    else if (result != vk::Result::eSuccess)
        throw std::runtime_error("Failed to acquire swapchain image");

    return _nextImage;
}

bool Swapchain::present(uint32_t waitSemaphoreCount, const vk::Semaphore* waitSemaphores)
{
    // TODO: noexcept, modern interface would throw on ErrorOutOfDate
    const vk::PresentInfoKHR presentInfo{
        waitSemaphoreCount,
        waitSemaphores,
        1, // swapchainCount
        &_swapchain,
        &_nextImage
    };
    const vk::Result result = _device->presentQueue().presentKHR(&presentInfo);

    // Swapchain should be recreated if out of date or suboptimal
    const bool good_swap = [&]{
        bool good_swap = true;
        if (result == vk::Result::eErrorOutOfDateKHR ||
            result == vk::Result::eSuboptimalKHR)
            good_swap = false;
        else if (result != vk::Result::eSuccess)
            throw std::runtime_error("Failed to present swapchain image");

        return good_swap;
    }();

    _nextFrame = (_nextFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    return good_swap;
}

void Swapchain::createSwapchain()
{
    const QueueFamilies indices = _device->queueFamilies();
    const std::array<uint32_t, 2> queueFamilyIndices = {{
        indices.graphicsFamily.value(),
        indices.presentFamily.value()
    }};

    // Handle ownership of images
    const auto [imageSharingMode, queueFamilyIndexCount, pQueueFamilyIndices] = [&] {
        if (indices.graphicsFamily != indices.presentFamily) {
            // Pick concurrent to skip in-depth ownership jazz for now
            return std::tuple<vk::SharingMode, uint32_t, const uint32_t*>(
                vk::SharingMode::eConcurrent,
                queueFamilyIndices.size(),
                queueFamilyIndices.data()
            );
        } else {
            return std::tuple<vk::SharingMode, uint32_t, const uint32_t*>(
                vk::SharingMode::eExclusive,
                0, // optional
                nullptr // optional
            );
        }
    }();

    _swapchain = _device->logical().createSwapchainKHR({
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
    });
}

void Swapchain::createImages()
{
    auto images =_device->logical().getSwapchainImagesKHR(_swapchain);
    for (auto& image : images) {
        _images.push_back({image, {}, {}});
        _images.back().view = _device->logical().createImageView({
            {}, // flags
            image,
            vk::ImageViewType::e2D,
            _config.surfaceFormat.format,
            {}, // Identity swizzles
            vk::ImageSubresourceRange{
                vk::ImageAspectFlagBits::eColor,
                0, // base mip
                1, // level count
                0, // base array layer
                1  // layer count
            }
        });
    }
}

void Swapchain::createDepthResources()
{
    // Check depth buffer without stencil is supported
    const auto features = vk::FormatFeatureFlagBits::eDepthStencilAttachment;
    const auto properties = _device->physical().getFormatProperties(_config.depthFormat);
    if ((properties.optimalTilingFeatures & features) != features)
        throw std::runtime_error("Depth format unsupported");

    const vk::ImageSubresourceRange subresourceRange{
        vk::ImageAspectFlagBits::eDepth,
        0, // baseMipLevel
        1, // levelCount
        0, // baseArrayLayer
        1 // layerCount
    };

    _depthImage = _device->createImage(
        _config.extent,
        _config.depthFormat,
        subresourceRange,
        vk::ImageViewType::e2D,
        vk::ImageTiling::eOptimal,
        vk::ImageCreateFlags{},
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    const auto commandBuffer = _device->beginGraphicsCommands();

    transitionImageLayout(
        _depthImage,
        subresourceRange,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthStencilAttachmentOptimal,
        commandBuffer
    );

    _device->endGraphicsCommands(commandBuffer);
}

void Swapchain::createFramebuffers(const vk::RenderPass renderPass)
{
    for (auto& image : _images) {
        const std::array<vk::ImageView, 2> attachments = {{
            image.view,
            _depthImage.view
        }};
        image.fbo = _device->logical().createFramebuffer({
            {}, // flags
            renderPass,
            attachments.size(),
            attachments.data(),
            _config.extent.width,
            _config.extent.height,
            1 // layers
        });
    }
}

void Swapchain::createFences()
{
    const vk::FenceCreateInfo fenceInfo(
        vk::FenceCreateFlagBits::eSignaled
    );
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        _inFlightFences.push_back(_device->logical().createFence(fenceInfo));
}

void Swapchain::destroy()
{
    if (_device) {
        for (auto fence : _inFlightFences)
            _device->logical().destroy(fence);
        _device->destroy(_depthImage);
        for (auto& image : _images) {
            _device->logical().destroy(image.fbo);
            _device->logical().destroy(image.view);
        }
        _device->logical().destroy(_swapchain);
    }

    _swapchain = vk::SwapchainKHR();
    _images.clear();
    _inFlightFences.clear();
    _nextFrame = 0;
    _nextImage = 0;
}
