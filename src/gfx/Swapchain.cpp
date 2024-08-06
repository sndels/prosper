#include "Swapchain.hpp"

#include "../utils/Logger.hpp"
#include "../utils/Utils.hpp"
#include "Device.hpp"
#include "VkUtils.hpp"

#include <stdexcept>
#include <wheels/containers/span.hpp>

using namespace wheels;

namespace
{

vk::SurfaceFormatKHR selectSwapSurfaceFormat(
    Span<const vk::SurfaceFormatKHR> availableFormats)
{
    if (availableFormats.size() == 1 &&
        availableFormats[0].format == vk::Format::eUndefined)
        // We're free to take our pick
        return {
            vk::Format::eB8G8R8A8Unorm,
            vk::ColorSpaceKHR::eSrgbNonlinear,
        };

    // Check if preferred sRGB format is present
    for (const auto &format : availableFormats)
    {
        bool const bgra8OrRgba8 = format.format == vk::Format::eB8G8R8A8Unorm ||
                                  format.format == vk::Format::eR8G8B8A8Unorm;
        bool const srgbNonlinear =
            format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
        if (bgra8OrRgba8 && srgbNonlinear)
            return format;
    }

    // At least one of the 8bit unorm surface formats is supported by rdna,
    // non-tegra nvidia and intel
    LOG_WARN(
        "Linear 8bit rgba surface not supported. Output might look incorrect.");

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
        {
            LOG_INFO("Using present mode 'Mailbox'");
            return mode;
        }
        // fifo is not properly supported by some drivers so use immediate if
        // available
        if (mode == vk::PresentModeKHR::eImmediate)
            bestMode = mode;
    }

    if (bestMode == vk::PresentModeKHR::eFifo)
        LOG_INFO("Using present mode 'Fifo'");
    else if (bestMode == vk::PresentModeKHR::eImmediate)
        LOG_INFO("Using present mode 'Immediate'");

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
    ScopedScratch scopeAlloc, const vk::Extent2D &preferredExtent)
{
    const SwapchainSupport support(
        scopeAlloc, gDevice.physical(), gDevice.surface());

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

Swapchain::~Swapchain() { destroy(); }

void Swapchain::init(const SwapchainConfig &config)
{
    WHEELS_ASSERT(!m_initialized);

    LOG_INFO("Creating Swapchain");

    recreate(config);

    m_initialized = true;
}

SwapchainConfig const &Swapchain::config() const
{
    WHEELS_ASSERT(m_initialized);

    return m_config;
}

vk::Format Swapchain::format() const
{
    WHEELS_ASSERT(m_initialized);

    return m_config.surfaceFormat.format;
}

const vk::Extent2D &Swapchain::extent() const
{
    WHEELS_ASSERT(m_initialized);

    return m_config.extent;
}

const SwapchainImage &Swapchain::image(size_t i) const
{
    WHEELS_ASSERT(m_initialized);

    if (i < m_images.size())
        return m_images[i];
    throw std::runtime_error("Tried to index past swap image count");
}

size_t Swapchain::nextFrame() const
{
    WHEELS_ASSERT(m_initialized);

    return m_nextFrame;
}

vk::Fence Swapchain::currentFence() const
{
    WHEELS_ASSERT(m_initialized);

    return m_inFlightFences[m_nextFrame];
}

wheels::Optional<uint32_t> Swapchain::acquireNextImage(
    vk::Semaphore signalSemaphore)
{
    WHEELS_ASSERT(m_initialized);

    const auto noTimeout = std::numeric_limits<uint64_t>::max();
    checkSuccess(
        gDevice.logical().waitForFences(
            1, &m_inFlightFences[m_nextFrame], VK_TRUE, noTimeout),
        "waitForFences");
    checkSuccess(
        gDevice.logical().resetFences(1, &m_inFlightFences[m_nextFrame]),
        "resetFences");

    // TODO: noexcept, modern interface would throw on ErrorOutOfDate
    const auto result = gDevice.logical().acquireNextImageKHR(
        m_swapchain, noTimeout, signalSemaphore, vk::Fence{}, &m_nextImage);
    WHEELS_ASSERT(m_nextImage < m_config.imageCount);

    // Swapchain should be recreated if out of date or suboptimal
    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR)
        return {};
    if (result != vk::Result::eSuccess)
        throw std::runtime_error("Failed to acquire swapchain image");

    return m_nextImage;
}

bool Swapchain::present(Span<const vk::Semaphore> waitSemaphores)
{
    WHEELS_ASSERT(m_initialized);

    // TODO: noexcept, modern interface would throw on ErrorOutOfDate
    const vk::PresentInfoKHR presentInfo{
        .waitSemaphoreCount = asserted_cast<uint32_t>(waitSemaphores.size()),
        .pWaitSemaphores = waitSemaphores.data(),
        .swapchainCount = 1,
        .pSwapchains = &m_swapchain,
        .pImageIndices = &m_nextImage,
    };
    const vk::Result result = gDevice.graphicsQueue().presentKHR(&presentInfo);

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

    m_nextFrame = (m_nextFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    return good_swap;
}

void Swapchain::recreate(const SwapchainConfig &config)
{
    // Called by init so no init assert
    destroy();
    m_config = config;
    createSwapchain();
    createImages();
    createFences();
}

void Swapchain::destroy()
{
    for (const vk::Fence f : m_inFlightFences)
        gDevice.logical().destroy(f);
    m_inFlightFences = {};
    m_images.clear();
    gDevice.logical().destroy(m_swapchain);
}

void Swapchain::createSwapchain()
{
    m_swapchain =
        gDevice.logical().createSwapchainKHR(vk::SwapchainCreateInfoKHR{
            .surface = gDevice.surface(),
            .minImageCount = m_config.imageCount,
            .imageFormat = m_config.surfaceFormat.format,
            .imageColorSpace = m_config.surfaceFormat.colorSpace,
            .imageExtent = m_config.extent,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eTransferDst,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = m_config.transform,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = m_config.presentMode,
            .clipped = VK_TRUE});
}

void Swapchain::createImages()
{
    auto images = gDevice.logical().getSwapchainImagesKHR(m_swapchain);
    for (auto &image : images)
    {
        m_images.push_back(SwapchainImage{
            .handle = image,
            .extent = m_config.extent,
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
    m_config.imageCount = asserted_cast<uint32_t>(m_images.size());
}

void Swapchain::createFences()
{
    const vk::FenceCreateInfo fenceInfo{
        .flags = vk::FenceCreateFlagBits::eSignaled,
    };
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        m_inFlightFences[i] = gDevice.logical().createFence(fenceInfo);
}
