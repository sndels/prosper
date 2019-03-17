#include "Device.hpp"

#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>

#include "App.hpp"
#include "Swapchain.hpp"

namespace {
    const std::vector<const char*> validationLayers = {
        //"VK_LAYER_LUNARG_api_dump",
        "VK_LAYER_LUNARG_standard_validation"
    };
    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    QueueFamilies findQueueFamilies(const vk::PhysicalDevice device, const vk::SurfaceKHR surface)
    {
        QueueFamilies families;

        // Get supported queue families
        const auto allFamilies = device.getQueueFamilyProperties();

        // Find needed queue support
        for (uint32_t i = 0; i < allFamilies.size(); ++i) {
            if (allFamilies[i].queueCount > 0) {
                // Query present support
                const vk::Bool32 presentSupport = device.getSurfaceSupportKHR(i, surface);

                // Set index to matching families 
                if (allFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics)
                    families.graphicsFamily = i;
                if (presentSupport)
                    families.presentFamily = i;
            }

            if (families.isComplete())
                break;
        }

        return families;
    }

    bool checkDeviceExtensionSupport(const vk::PhysicalDevice device)
    {
        // Find out available extensions
        const auto availableExtensions = device.enumerateDeviceExtensionProperties(nullptr);

        // Check that all needed extensions are present
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    bool checkValidationLayerSupport()
    {
        const auto availableLayers = vk::enumerateInstanceLayerProperties();

        // Check that each layer is supported
        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    std::vector<const char*> getRequiredExtensions()
    {
        // Query extensions glfw requires
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        // Add extension containing debug layers
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        return extensions;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        const VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        const VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData)
    {
        (void) messageSeverity;
        (void) messageType;
        (void) pUserData;

        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE; // Don't fail the causing command
    }

    uint32_t findMemoryType(const vk::PhysicalDevice physical, const uint32_t typeFilter, const vk::MemoryPropertyFlags properties)
    {
        const auto memProperties = physical.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
            if(typeFilter & (1 << i) &&
               (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type");
    }

    void CreateDebugUtilsMessengerEXT(
        const vk::Instance instance,
        const vk::DebugUtilsMessengerCreateInfoEXT* pCreateInfo,
        const vk::AllocationCallbacks* pAllocator,
        vk::DebugUtilsMessengerEXT* pDebugMessenger)
    {
        auto vkpCreateInfo = reinterpret_cast<const VkDebugUtilsMessengerCreateInfoEXT*>(pCreateInfo);
        auto vkpAllocator = reinterpret_cast<const VkAllocationCallbacks*>(pAllocator);
        auto vkpDebugMessenger = reinterpret_cast<VkDebugUtilsMessengerEXT*>(pDebugMessenger);

        auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (func == nullptr || 
            func(instance, vkpCreateInfo, vkpAllocator, vkpDebugMessenger) != VK_SUCCESS)
            throw std::runtime_error("failed to create debug messenger");
    }

    void DestroyDebugUtilsMessengerEXT(
        const vk::Instance instance,
        const vk::DebugUtilsMessengerEXT debugMessenger,
        const vk::AllocationCallbacks* pAllocator)
     {
        auto vkpAllocator = reinterpret_cast<const VkAllocationCallbacks*>(pAllocator);

        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr)
            func(instance, debugMessenger, vkpAllocator);
    }

    std::pair<vk::AccessFlags, vk::AccessFlags> accessMasks(const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout)
    {
        if (oldLayout == vk::ImageLayout::eUndefined &&
            newLayout == vk::ImageLayout::eTransferDstOptimal) {
            return std::pair(
                vk::AccessFlags(),
                vk::AccessFlagBits::eTransferWrite
            );
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
                   newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            return std::pair(
                vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eShaderRead
            );
        } else
            throw std::runtime_error("Unsupported layout transition");
    }

    std::pair<vk::PipelineStageFlags, vk::PipelineStageFlags> stageMasks(const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout)
    {
        if (oldLayout == vk::ImageLayout::eUndefined &&
            newLayout == vk::ImageLayout::eTransferDstOptimal) {
            return std::pair(
                vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eTransfer
            );
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
                   newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            return std::pair(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eFragmentShader
            );
        } else
            throw std::runtime_error("Unsupported layout transition");
    }
}

Device::~Device()
{
    // Also cleans up associated command buffers
    _logical.destroy(_commandPool);
    // Implicitly cleans up associated queues as well
    _logical.destroy();
    _instance.destroy(_surface);
    DestroyDebugUtilsMessengerEXT(_instance, _debugMessenger, nullptr);
    _instance.destroy();
}

void Device::init(GLFWwindow* window)
{
    createInstance();
    createDebugMessenger();
    createSurface(window);
    selectPhysicalDevice();
    _queueFamilies = findQueueFamilies(_physical, _surface);
    createLogicalDevice();
    createCommandPool();
}

vk::Instance Device::instance() const
{
    return _instance;
}

vk::PhysicalDevice Device::physical() const
{
    return _physical;
}

vk::Device Device::logical() const
{
    return _logical;
}

vk::SurfaceKHR Device::surface() const
{
    return _surface;
}

vk::CommandPool Device::commandPool() const
{
    return _commandPool;
}

vk::Queue Device::graphicsQueue() const
{
    return _graphicsQueue;
}

vk::Queue Device::presentQueue() const
{
    return _presentQueue;
}

const QueueFamilies& Device::queueFamilies() const
{
    return _queueFamilies;
}

Buffer Device::createBuffer(const vk::DeviceSize size, const vk::BufferUsageFlags usage, const vk::MemoryPropertyFlags properties) const
{
    Buffer buffer;

    // Create the buffer
    const vk::BufferCreateInfo bufferInfo(
        {}, // flags
        size,
        usage,
        vk::SharingMode::eExclusive
    );
    buffer.handle = _logical.createBuffer(bufferInfo);

    // Check memory requirements
    const auto memRequirements = _logical.getBufferMemoryRequirements(buffer.handle);

    // Allocate memory for it
    const vk::MemoryAllocateInfo allocInfo(
        memRequirements.size, 
        findMemoryType(_physical, memRequirements.memoryTypeBits, properties)
    );
    buffer.memory = _logical.allocateMemory(allocInfo);

    // Bind memory to buffer
    _logical.bindBufferMemory(buffer.handle, buffer.memory, 0);

    return buffer;
}

void Device::copyBuffer(const Buffer& src, const Buffer& dst, const vk::DeviceSize size) const
{
    const auto commandBuffer = beginGraphicsCommands();

    const vk::BufferCopy copyRegion(
        0, // srcOffset
        0, // dstOffset
        size
    );
    commandBuffer.copyBuffer(src.handle, dst.handle, 1, &copyRegion);

    endGraphicsCommands(commandBuffer);
}

void Device::copyBufferToImage(const Buffer& src, const Image& dst, const vk::Extent2D extent) const
{
    const auto commandBuffer = beginGraphicsCommands();

    const vk::BufferImageCopy region(
        0, // bufferOffset
        0, // bufferRowLength
        0, // bufferImageHeight
        vk::ImageSubresourceLayers(
            vk::ImageAspectFlagBits::eColor,
            0, // mipLevel
            0, // arrayLayer
            1 // layerCount
        ),
        vk::Offset3D(0, 0, 0),
        vk::Extent3D(extent, 1)
    );
    commandBuffer.copyBufferToImage(
        src.handle,
        dst.handle,
        vk::ImageLayout::eTransferDstOptimal,
        1, // regionCount
        &region
    );

    endGraphicsCommands(commandBuffer);
}

Image Device::createImage(const vk::Extent2D extent, const vk::Format format, const vk::ImageTiling tiling, const vk::ImageUsageFlags usage, const vk::MemoryPropertyFlags properties) const
{
    Image image;

    // Create handle
    const vk::ImageCreateInfo imageInfo(
        {}, // flags
        vk::ImageType::e2D,
        vk::Format::eR8G8B8A8Unorm,
        vk::Extent3D(extent, 1),
        1, // mipLevels
        1, // arrayLayers
        vk::SampleCountFlagBits::e1,
        tiling,
        usage,
        vk::SharingMode::eExclusive
    );
    image.handle = _logical.createImage(imageInfo);

    // Allocate and bind memory
    const auto memRequirements = _logical.getImageMemoryRequirements(image.handle);
    const vk::MemoryAllocateInfo allocInfo(
        memRequirements.size,
        findMemoryType(_physical, memRequirements.memoryTypeBits, properties)
    );
    image.memory = _logical.allocateMemory(allocInfo);
    _logical.bindImageMemory(image.handle, image.memory, 0);

    return image;
}

void Device::transitionImageLayout(const Image& image, const vk::Format format, const vk::ImageSubresourceRange& subresourceRange, const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout) const
{
    const auto commandBuffer = beginGraphicsCommands();

    const auto [srcAccessMask, dstAccessMask] = accessMasks(oldLayout, newLayout);
    const auto [srcStageMask, dstStageMask] = stageMasks(oldLayout, newLayout);

    const vk::ImageMemoryBarrier barrier(
        srcAccessMask,
        dstAccessMask,
        oldLayout,
        newLayout,
        VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
        VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
        image.handle,
        subresourceRange
    );
    commandBuffer.pipelineBarrier(
        srcStageMask,
        dstStageMask,
        {}, // dependencyFlags
        0, nullptr, // memoryBarriers
        0, nullptr, // bufferMemoryBarriers
        1, &barrier
    );

    endGraphicsCommands(commandBuffer);
}

vk::CommandBuffer Device::beginGraphicsCommands() const
{
    // Allocate and begin a command buffer
    const vk::CommandBufferAllocateInfo allocInfo(
        _commandPool,
        vk::CommandBufferLevel::ePrimary,
        1 // commandBufferCount
    );
    const auto buffer = _logical.allocateCommandBuffers(allocInfo)[0];

    const vk::CommandBufferBeginInfo beginInfo(
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit
    );
    buffer.begin(beginInfo);

    return buffer;
}

void Device::endGraphicsCommands(const vk::CommandBuffer buffer) const
{
    // End and submit on graphics queue
    buffer.end();

    const vk::SubmitInfo submitInfo(
        0, // waitSemaphoreCount
        nullptr, // pWaitSemaphores
        nullptr, // pWaitDstStageMask
        1, // commandBufferCount
        &buffer
    );
    _graphicsQueue.submit(1, &submitInfo, {});
    _graphicsQueue.waitIdle(); // TODO: Collect setup commands and execute at once

    _logical.freeCommandBuffers(_commandPool, 1, &buffer);
}

bool Device::isDeviceSuitable(const vk::PhysicalDevice device) const
{
    const auto families = findQueueFamilies(device, _surface);

    const auto extensionsSupported = checkDeviceExtensionSupport(device);
    const auto supportedFeatures = device.getFeatures();

    const bool swapChainAdequate = [&]{
        bool adequate = false;
        if (extensionsSupported) {
            SwapchainSupport swapSupport = querySwapchainSupport(device, _surface);
            adequate = !swapSupport.formats.empty() && !swapSupport.presentModes.empty();
        }

        return adequate;
    }();

    return families.isComplete() &&
           extensionsSupported &&
           swapChainAdequate &&
           supportedFeatures.samplerAnisotropy;
}

void Device::createInstance()
{
    if (!checkValidationLayerSupport())
        throw std::runtime_error("Validation layers not available");

    // Setup app info
    const vk::ApplicationInfo appInfo(
        "prosper",
        VK_MAKE_VERSION(1, 0, 0),
        "prosper",
        VK_MAKE_VERSION(1, 0, 0),
        VK_API_VERSION_1_0
    );

    // Gather required extensions
    const auto extensions = getRequiredExtensions();

    // Create instance
    const vk::InstanceCreateInfo createInfo(
        {}, // flags
        &appInfo,
        validationLayers.size(),
        validationLayers.data(),
        extensions.size(),
        extensions.data()
    );
    _instance = vk::createInstance(createInfo);
}

void Device::createDebugMessenger()
{
    // Create debug messenger with everything except info
    const vk::DebugUtilsMessengerCreateInfoEXT createInfo(
        {},
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        debugCallback
    );
    CreateDebugUtilsMessengerEXT(_instance, &createInfo, nullptr, &_debugMessenger);
}

void Device::createSurface(GLFWwindow* window)
{
    // TODO: Seems legit cast?
    auto vkpSurface = reinterpret_cast<VkSurfaceKHR*>(&_surface);
    if (glfwCreateWindowSurface(_instance, window, nullptr, vkpSurface) != VK_SUCCESS)
        throw std::runtime_error("Failed to create window surface");
}

void Device::selectPhysicalDevice()
{
    // Find physical devices
    const auto devices = _instance.enumeratePhysicalDevices();

    // Select a suitable one
    for (const auto& device : devices) {
        if (isDeviceSuitable(device)) {
            _physical = device;
            return;
        }
    }

    throw std::runtime_error("Failed to find a suitable GPU");
}

void Device::createLogicalDevice()
{
    const uint32_t graphicsFamily = _queueFamilies.graphicsFamily.value();
    const uint32_t presentFamily = _queueFamilies.presentFamily.value();

    // Config queues, concatenating duplicate families
    const float queuePriority = 1;
    const std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos = [&]{
        std::vector<vk::DeviceQueueCreateInfo> cis;
        const std::set<uint32_t> uniqueQueueFamilies = {
            graphicsFamily,
            presentFamily
        };
        for (uint32_t family : uniqueQueueFamilies) {
            const vk::DeviceQueueCreateInfo queueCreateInfo(
                {}, //flags
                family,
                1, // queueCount
                &queuePriority
            );
            cis.push_back(queueCreateInfo);
        }
        return cis;
    }();

    // Set up features
    const vk::PhysicalDeviceFeatures deviceFeatures = [&]{
        vk::PhysicalDeviceFeatures features;
        features.samplerAnisotropy = VK_TRUE;

        return features;
    }();

    // Create logical device
    const vk::DeviceCreateInfo createInfo(
        {}, // flags
        queueCreateInfos.size(),
        queueCreateInfos.data(),
        validationLayers.size(),
        validationLayers.data(),
        deviceExtensions.size(),
        deviceExtensions.data(),
        &deviceFeatures
    );
    _logical = _physical.createDevice(createInfo);

    // Get the created queues
    _graphicsQueue = _logical.getQueue(graphicsFamily, 0);
    _presentQueue = _logical.getQueue(presentFamily, 0);
}

void Device::createCommandPool()
{
    const vk::CommandPoolCreateInfo poolInfo(
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        _queueFamilies.graphicsFamily.value()
    );
    _commandPool = _logical.createCommandPool(poolInfo, nullptr);
}
