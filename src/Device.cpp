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
        "VK_LAYER_KHRONOS_validation"
    };
    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    QueueFamilies findQueueFamilies(const vk::PhysicalDevice device, const vk::SurfaceKHR surface)
    {
        const auto allFamilies = device.getQueueFamilyProperties();

        // Find needed queue support
        QueueFamilies families;
        for (uint32_t i = 0; i < allFamilies.size(); ++i) {
            if (allFamilies[i].queueCount > 0) {
                const vk::Bool32 presentSupport = device.getSurfaceSupportKHR(i, surface);

                // Set index to matching families
                if (allFamilies[i].queueFlags & vk::QueueFlagBits::eCompute)
                    families.computeFamily = i;
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

    void CreateDebugUtilsMessengerEXT(
        const vk::Instance instance,
        const vk::DebugUtilsMessengerCreateInfoEXT* pCreateInfo,
        const vk::AllocationCallbacks* pAllocator,
        vk::DebugUtilsMessengerEXT* pDebugMessenger)
    {
        auto vkInstance = static_cast<VkInstance>(instance);
        auto vkpCreateInfo = reinterpret_cast<const VkDebugUtilsMessengerCreateInfoEXT*>(pCreateInfo);
        auto vkpAllocator = reinterpret_cast<const VkAllocationCallbacks*>(pAllocator);
        auto vkpDebugMessenger = reinterpret_cast<VkDebugUtilsMessengerEXT*>(pDebugMessenger);

        auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(vkInstance, "vkCreateDebugUtilsMessengerEXT");
        if (func == nullptr ||
            func(vkInstance, vkpCreateInfo, vkpAllocator, vkpDebugMessenger) != VK_SUCCESS)
            throw std::runtime_error("failed to create debug messenger");
    }

    void DestroyDebugUtilsMessengerEXT(
        const vk::Instance instance,
        const vk::DebugUtilsMessengerEXT debugMessenger,
        const vk::AllocationCallbacks* pAllocator)
     {
        auto vkInstance = static_cast<VkInstance>(instance);
        const auto vkDebugMessenger = static_cast<const VkDebugUtilsMessengerEXT>(debugMessenger);
        auto vkpAllocator = reinterpret_cast<const VkAllocationCallbacks*>(pAllocator);

        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(vkInstance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr)
            func(vkInstance, vkDebugMessenger, vkpAllocator);
    }
}

Device::Device(GLFWwindow* window)
{
    createInstance();
    createDebugMessenger();
    createSurface(window);
    selectPhysicalDevice();
    _queueFamilies = findQueueFamilies(_physical, _surface);
    createLogicalDevice();
    createAllocator();
    createCommandPools();
}

Device::~Device()
{
    // Also cleans up associated command buffers
    _logical.destroy(_graphicsPool);
    _logical.destroy(_computePool);
    vmaDestroyAllocator(_allocator);
    // Implicitly cleans up associated queues as well
    _logical.destroy();
    _instance.destroy(_surface);
    DestroyDebugUtilsMessengerEXT(_instance, _debugMessenger, nullptr);
    _instance.destroy();
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

vk::CommandPool Device::computePool() const
{
    return _graphicsPool;
}

vk::CommandPool Device::graphicsPool() const
{
    return _graphicsPool;
}

vk::Queue Device::computeQueue() const
{
    return _computeQueue;
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

void Device::map(const VmaAllocation allocation, void** data) const
{
    vmaMapMemory(_allocator, allocation, data);
}

void Device::unmap(const VmaAllocation allocation) const
{
    vmaUnmapMemory(_allocator, allocation);
}

Buffer Device::createBuffer(const vk::DeviceSize size, const vk::BufferUsageFlags usage, const vk::MemoryPropertyFlags properties, const VmaMemoryUsage vmaUsage) const
{
    vk::BufferCreateInfo bufferInfo{
        {}, // flags
        size,
        usage,
        vk::SharingMode::eExclusive
    };
    // TODO: preferred flags, create mapped
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = vmaUsage;
    allocInfo.requiredFlags = static_cast<VkMemoryPropertyFlags>(properties);

    Buffer buffer;
    auto vkpBufferInfo = reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo);
    auto vkpBuffer = reinterpret_cast<VkBuffer*>(&buffer.handle);
    vmaCreateBuffer(
        _allocator,
        vkpBufferInfo,
        &allocInfo,
        vkpBuffer,
        &buffer.allocation,
        nullptr
    );
    return buffer;
}

void Device::destroy(const Buffer& buffer) const
{
    const auto vkBuffer = static_cast<VkBuffer>(buffer.handle);
    vmaDestroyBuffer(_allocator, vkBuffer, buffer.allocation);
}

Image Device::createImage(const vk::Extent2D extent, const vk::Format format, const vk::ImageSubresourceRange& range, const vk::ImageViewType viewType, const vk::ImageTiling tiling, const vk::ImageCreateFlags flags, const vk::ImageUsageFlags usage, const vk::MemoryPropertyFlags properties, const VmaMemoryUsage vmaUsage) const
{
    vk::ImageCreateInfo imageInfo{
        flags,
        vk::ImageType::e2D,
        format,
        vk::Extent3D{extent, 1},
        range.levelCount,
        range.layerCount,
        vk::SampleCountFlagBits::e1,
        tiling,
        usage,
        vk::SharingMode::eExclusive
    };
    // TODO: preferred flags, create mapped
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = vmaUsage;
    allocInfo.requiredFlags = static_cast<VkMemoryPropertyFlags>(properties);

    Image image;
    auto vkpImageInfo = reinterpret_cast<VkImageCreateInfo*>(&imageInfo);
    auto vkpImage = reinterpret_cast<VkImage*>(&image.handle);
    vmaCreateImage(
        _allocator,
        vkpImageInfo,
        &allocInfo,
        vkpImage,
        &image.allocation,
        nullptr
    );

    image.view = _logical.createImageView({
        {}, // flags
        image.handle,
        viewType,
        format,
        vk::ComponentMapping{},
        range
    });

    image.format = format;
    return image;
}

void Device::destroy(const Image& image) const
{
    const auto vkImage = static_cast<VkImage>(image.handle);
    vmaDestroyImage(_allocator, vkImage, image.allocation);
    _logical.destroy(image.view);
}

vk::CommandBuffer Device::beginGraphicsCommands() const
{
    const auto buffer = _logical.allocateCommandBuffers({
        _graphicsPool,
        vk::CommandBufferLevel::ePrimary,
        1 // commandBufferCount
    })[0];

    const vk::CommandBufferBeginInfo beginInfo(
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit
    );
    buffer.begin(beginInfo);

    return buffer;
}

void Device::endGraphicsCommands(const vk::CommandBuffer buffer) const
{
    buffer.end();

    const vk::SubmitInfo submitInfo{
        0, // waitSemaphoreCount
        nullptr, // pWaitSemaphores
        nullptr, // pWaitDstStageMask
        1, // commandBufferCount
        &buffer
    };
    _graphicsQueue.submit(1, &submitInfo, {});
    _graphicsQueue.waitIdle(); // TODO: Collect setup commands and execute at once

    _logical.freeCommandBuffers(_graphicsPool, 1, &buffer);
}

bool Device::isDeviceSuitable(const vk::PhysicalDevice device) const
{
    const auto families = findQueueFamilies(device, _surface);

    const auto extensionsSupported = checkDeviceExtensionSupport(device);
    const auto supportedFeatures = device.getFeatures();

    const bool swapChainAdequate = [&]{
        bool adequate = false;
        if (extensionsSupported) {
            SwapchainSupport swapSupport{device, _surface};
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

    const vk::ApplicationInfo appInfo{
        "prosper",
        VK_MAKE_VERSION(1, 0, 0),
        "prosper",
        VK_MAKE_VERSION(1, 0, 0),
        VK_API_VERSION_1_0
    };

    const auto extensions = getRequiredExtensions();

    _instance = vk::createInstance({
        {}, // flags
        &appInfo,
        static_cast<uint32_t>(validationLayers.size()),
        validationLayers.data(),
        static_cast<uint32_t>(extensions.size()),
        extensions.data()
    });
}

void Device::createDebugMessenger()
{
    const vk::DebugUtilsMessengerCreateInfoEXT createInfo{
        {},
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        debugCallback
    };
    CreateDebugUtilsMessengerEXT(_instance, &createInfo, nullptr, &_debugMessenger);
}

void Device::createSurface(GLFWwindow* window)
{
    auto vkpSurface = reinterpret_cast<VkSurfaceKHR*>(&_surface);
    auto vkInstance = static_cast<VkInstance>(_instance);
    if (glfwCreateWindowSurface(vkInstance, window, nullptr, vkpSurface) != VK_SUCCESS)
        throw std::runtime_error("Failed to create window surface");
}

void Device::selectPhysicalDevice()
{
    const auto devices = _instance.enumeratePhysicalDevices();

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
    const uint32_t computeFamily = _queueFamilies.computeFamily.value();
    const uint32_t graphicsFamily = _queueFamilies.graphicsFamily.value();
    const uint32_t presentFamily = _queueFamilies.presentFamily.value();

    // Config queues, concat duplicate families
    const float queuePriority = 1;
    const std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos = [&]{
        std::vector<vk::DeviceQueueCreateInfo> cis;
        const std::set<uint32_t> uniqueQueueFamilies = {
            computeFamily,
            graphicsFamily,
            presentFamily
        };
        for (uint32_t family : uniqueQueueFamilies) {
            cis.push_back({
                {}, //flags
                family,
                1, // queueCount
                &queuePriority
            });
        }
        return cis;
    }();

    const vk::PhysicalDeviceFeatures deviceFeatures = [&]{
        vk::PhysicalDeviceFeatures features;
        features.samplerAnisotropy = VK_TRUE;

        return features;
    }();

    _logical = _physical.createDevice({
        {}, // flags
        static_cast<uint32_t>(queueCreateInfos.size()),
        queueCreateInfos.data(),
        static_cast<uint32_t>(validationLayers.size()),
        validationLayers.data(),
        static_cast<uint32_t>(deviceExtensions.size()),
        deviceExtensions.data(),
        &deviceFeatures
    });

    // Get the created queues
    _computeQueue = _logical.getQueue(computeFamily, 0);
    _graphicsQueue = _logical.getQueue(graphicsFamily, 0);
    _presentQueue = _logical.getQueue(presentFamily, 0);
}

void Device::createAllocator()
{
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = static_cast<VkPhysicalDevice>(_physical);
    allocatorInfo.device = static_cast<VkDevice>(_logical);
    if (vmaCreateAllocator(&allocatorInfo, &_allocator) != VK_SUCCESS)
        throw std::runtime_error("Failed to create allocator");
}

void Device::createCommandPools()
{
    {
        const vk::CommandPoolCreateInfo poolInfo(
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            _queueFamilies.graphicsFamily.value()
        );
        _graphicsPool = _logical.createCommandPool(poolInfo, nullptr);
    }

    {
        const vk::CommandPoolCreateInfo poolInfo(
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            _queueFamilies.computeFamily.value()
        );
        _computePool = _logical.createCommandPool(poolInfo, nullptr);
    }
}
