#include "Swapchain.hpp"

#include <stdexcept>
#include <wheels/containers/span.hpp>

#include "../utils/Utils.hpp"
#include "Device.hpp"
#include "VkUtils.hpp"

using namespace wheels;

namespace
{

vk::SurfaceFormatKHR selectSwapSurfaceFormat(
    Span<const vk::SurfaceFormatKHR> availableFormats)
{
    // We're free to take our pick (sRGB output with "regular" 8bit rgba buffer)
    if (availableFormats.size() == 1 &&
        availableFormats[0].format == vk::Format::eUndefined)
        return {
            vk::Format::eB8G8R8A8Unorm,
            vk::ColorSpaceKHR::eSrgbNonlinear,
        };

    // Check if preferred sRGB format is present
    for (const auto &format : availableFormats)
    {
        if (format.format == vk::Format::eB8G8R8A8Unorm &&
            format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            return format;
    }

    // Default to the first one if preferred was not present
    // Picking "best one" is also an option here
    return availableFormats[0];
}

vk::PresentModeKHR selectSwapPresentMode(
    Span<const vk::PresentModeKHR> availablePresentModes)
{
    // Default to fifo (double buffering)
    vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;

    for (const auto &mode : availablePresentModes)
    {
        // We'd like mailbox to implement triple buffering
        if (mode == vk::PresentModeKHR::eMailbox)
            return mode;
        // fifo is not properly supported by some drivers so use immediate if
        // available
        if (mode == vk::PresentModeKHR::eImmediate)
            bestMode = mode;
    }

    return bestMode;
}

constexpr vk::Extent2D selectSwapExtent(
    const vk::Extent2D &extent, const vk::SurfaceCapabilitiesKHR &capabilities)
{
    // Check if we have a fixed extent
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max())
        return capabilities.currentExtent;

    // Pick best resolution from given bounds
    const vk::Extent2D actualExtent{
        std::clamp(
            extent.width, capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width),
        std::clamp(
            extent.height, capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height),
    };

    return actualExtent;
}

} // namespace

SwapchainSupport::SwapchainSupport(
    Allocator &alloc, vk::PhysicalDevice device, const vk::SurfaceKHR surface)
: capabilities{device.getSurfaceCapabilitiesKHR(surface)}
, formats{alloc}
, presentModes{alloc}
{
    {
        uint32_t count = 0;
        checkSuccess(
            device.getSurfaceFormatsKHR(surface, &count, nullptr),
            "Failed to get surface format count");

        formats.resize(count);
        checkSuccess(
            device.getSurfaceFormatsKHR(surface, &count, formats.data()),
            "Failed to get surface formats");
    }

    {
        uint32_t count = 0;
        checkSuccess(
            device.getSurfacePresentModesKHR(surface, &count, nullptr),
            "Failed to get present mode count");

        presentModes.resize(count);
        checkSuccess(
            device.getSurfacePresentModesKHR(
                surface, &count, presentModes.data()),
            "Failed to get present modes");
    }
}

SwapchainConfig::SwapchainConfig(
    ScopedScratch scopeAlloc, Device *device,
    const vk::Extent2D &preferredExtent)
{
    WHEELS_ASSERT(device != nullptr);

    const SwapchainSupport support(
        scopeAlloc, device->physical(), device->surface());

    // Needed to blit into, not supported by all implementations
    if (!(support.capabilities.supportedUsageFlags &
          vk::ImageUsageFlagBits::eTransferDst))
        throw std::runtime_error(
            "TransferDst usage not supported by swap surface");

    transform = support.capabilities.currentTransform;
    surfaceFormat = selectSwapSurfaceFormat(support.formats);
    presentMode = selectSwapPresentMode(support.presentModes);
    extent = selectSwapExtent(preferredExtent, support.capabilities);
    imageCount =
        support.capabilities.minImageCount +
        1; // Prefer one extra image to limit waiting on internal operations

    if (support.capabilities.maxImageCount > 0 &&
        imageCount > support.capabilities.maxImageCount)
        imageCount = support.capabilities.maxImageCount;
}

Swapchain::Swapchain(Device *device, const SwapchainConfig &config)
: _device{device}
, _config{config}
{
    WHEELS_ASSERT(_device != nullptr);

    printf("Creating Swapchain\n");

    recreate(config);
}

Swapchain::~Swapchain() { destroy(); }

SwapchainConfig const &Swapchain::config() const { return _config; }

vk::Format Swapchain::format() const { return _config.surfaceFormat.format; }

const vk::Extent2D &Swapchain::extent() const { return _config.extent; }

SwapchainImage Swapchain::image(size_t i) const
{
    if (i < _images.size())
        return _images[i];
    throw std::runtime_error("Tried to index past swap image count");
}

size_t Swapchain::nextFrame() const { return _nextFrame; }

vk::Fence Swapchain::currentFence() const
{
    return _inFlightFences[_nextFrame];
}

wheels::Optional<uint32_t> Swapchain::acquireNextImage(
    vk::Semaphore signalSemaphore)
{
    const auto noTimeout = std::numeric_limits<uint64_t>::max();
    checkSuccess(
        _device->logical().waitForFences(
            1, &_inFlightFences[_nextFrame], VK_TRUE, noTimeout),
        "waitForFences");
    checkSuccess(
        _device->logical().resetFences(1, &_inFlightFences[_nextFrame]),
        "resetFences");

    // TODO: noexcept, modern interface would throw on ErrorOutOfDate
    const auto result = _device->logical().acquireNextImageKHR(
        _swapchain, noTimeout, signalSemaphore, vk::Fence{}, &_nextImage);
    WHEELS_ASSERT(_nextImage < _config.imageCount);

    // Swapchain should be recreated if out of date or suboptimal
    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR)
        return {};
    if (result != vk::Result::eSuccess)
        throw std::runtime_error("Failed to acquire swapchain image");

    return _nextImage;
}

bool Swapchain::present(Span<const vk::Semaphore> waitSemaphores)
{
    // TODO: noexcept, modern interface would throw on ErrorOutOfDate
    const vk::PresentInfoKHR presentInfo{
        .waitSemaphoreCount = asserted_cast<uint32_t>(waitSemaphores.size()),
        .pWaitSemaphores = waitSemaphores.data(),
        .swapchainCount = 1,
        .pSwapchains = &_swapchain,
        .pImageIndices = &_nextImage,
    };
    const vk::Result result = _device->graphicsQueue().presentKHR(&presentInfo);

    // Swapchain should be recreated if out of date or suboptimal
    const bool good_swap = [&]
    {
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

void Swapchain::recreate(const SwapchainConfig &config)
{
    destroy();
    _config = config;
    createSwapchain();
    createImages();
    createFences();
}

void Swapchain::destroy()
{
    if (_device != nullptr)
    {
        for (auto fence : _inFlightFences)
            _device->logical().destroy(fence);
        _inFlightFences.clear();
        _images.clear();
        _device->logical().destroy(_swapchain);
    }
}

void Swapchain::createSwapchain()
{
    _swapchain =
        _device->logical().createSwapchainKHR(vk::SwapchainCreateInfoKHR{
            .surface = _device->surface(),
            .minImageCount = _config.imageCount,
            .imageFormat = _config.surfaceFormat.format,
            .imageColorSpace = _config.surfaceFormat.colorSpace,
            .imageExtent = _config.extent,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eTransferDst,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = _config.transform,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = _config.presentMode,
            .clipped = VK_TRUE});
}

void Swapchain::createImages()
{
    auto images = _device->logical().getSwapchainImagesKHR(_swapchain);
    for (auto &image : images)
    {
        _images.push_back(SwapchainImage{
            .handle = image,
            .extent = _config.extent,
            .subresourceRange =
                vk::ImageSubresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        });
    }
    // We might get more images than we asked for and acquire will use them all
    _config.imageCount = asserted_cast<uint32_t>(_images.size());
}

void Swapchain::createFences()
{
    const vk::FenceCreateInfo fenceInfo{
        .flags = vk::FenceCreateFlagBits::eSignaled,
    };
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        _inFlightFences.push_back(_device->logical().createFence(fenceInfo));
}
