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

    QueueFamilies findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface)
    {
        QueueFamilies families;

        // Get supported queue families
        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> allFamilies(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, allFamilies.data());

        // Find needed queue support
        for (uint32_t i = 0; i < allFamilies.size(); ++i) {
            if (allFamilies[i].queueCount > 0) {
                // Query present support
                VkBool32 presentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

                // Set index to matching families 
                if (allFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                    families.graphicsFamily = i;
                if (presentSupport)
                    families.presentFamily = i;
            }

            if (families.isComplete())
                break;
        }

        return families;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device)
    {
        // Find out available extensions
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        // Check that all needed extensions are present
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    bool checkValidationLayerSupport()
    {
        // Get supported layer count
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        // Get supported layer data
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

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
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData)
    {
        (void) messageSeverity;
        (void) messageType;
        (void) pUserData;

        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE; // Don't fail the causing command
    }

    VkResult CreateDebugUtilsMessengerEXT(
        VkInstance instance,
        const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
        const VkAllocationCallbacks* pAllocator,
        VkDebugUtilsMessengerEXT* pDebugMessenger)
    {
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (func == nullptr)
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        else
            return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }

    void DestroyDebugUtilsMessengerEXT(
        VkInstance instance,
        VkDebugUtilsMessengerEXT debugMessenger,
        const VkAllocationCallbacks* pAllocator)
    {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr)
            func(instance, debugMessenger, pAllocator);
    }

    uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
            if(typeFilter & (1 << i) &&
               (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type");
    }
}

Device::~Device()
{
    // Also cleans up associated command buffers
    vkDestroyCommandPool(_device, _commandPool, nullptr);
    // Implicitly cleans up associated queues as well
    vkDestroyDevice(_device, nullptr);
    vkDestroySurfaceKHR(_instance, _surface, nullptr);
    DestroyDebugUtilsMessengerEXT(_instance, _debugMessenger, nullptr);
    vkDestroyInstance(_instance, nullptr);
}

void Device::init(GLFWwindow* window)
{
    createInstance();
    createDebugMessenger();
    createSurface(window);
    selectPhysicalDevice();
    _queueFamilies = findQueueFamilies(_physicalDevice, _surface);
    createLogicalDevice();
    createCommandPool();
}

VkInstance Device::instance()
{
    return _instance;
}

VkPhysicalDevice Device::physicalDevice()
{
    return _physicalDevice;
}

VkDevice Device::handle()
{
    return _device;
}

VkSurfaceKHR Device::surface()
{
    return _surface;
}

VkCommandPool Device::commandPool()
{
    return _commandPool;
}

VkQueue Device::graphicsQueue()
{
    return _graphicsQueue;
}

VkQueue Device::presentQueue()
{
    return _presentQueue;
}

const QueueFamilies& Device::queueFamilies() const
{
    return _queueFamilies;
}

void Device::createBuffer(VkBuffer* buffer, VkDeviceMemory* bufferMemory, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties)
{
    // Create the buffer
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(_device, &bufferInfo, nullptr, buffer) != VK_SUCCESS)
        throw std::runtime_error("Failed to create vertex buffer");

    // Allocate memory for it
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(_device, *buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize =  memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(_physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(_device, &allocInfo, nullptr, bufferMemory) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate vertex buffer memory");

    // Bind memory to buffer
    vkBindBufferMemory(_device, *buffer, *bufferMemory, 0);
}

void Device::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size)
{
    // Allocate command buffer for copy operation
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = _commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(_device, &allocInfo, &commandBuffer);

    // Record copy
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0; // optional
    copyRegion.dstOffset = 0; // optional
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, src, dst, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    // Execute copy and wait for completion
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(_graphicsQueue);

    // Clean up
    vkFreeCommandBuffers(_device, _commandPool, 1, &commandBuffer);
}

bool Device::isDeviceSuitable(VkPhysicalDevice device)
{
    QueueFamilies families = findQueueFamilies(device, _surface);

    bool extensionsSupported = checkDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
        SwapchainSupport swapSupport = querySwapchainSupport(device, _surface);
        swapChainAdequate = !swapSupport.formats.empty() && !swapSupport.presentModes.empty();
    }

    return families.isComplete() && extensionsSupported && swapChainAdequate;
}

void Device::createInstance()
{
    if (!checkValidationLayerSupport())
        throw std::runtime_error("Validation layers not available");

    // Setup app info
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "prosper";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "prosper";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    // Gather required extensions
    std::vector<const char*> extensions = getRequiredExtensions();

    // Setup instance info
    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = extensions.size();
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();

    // Create instance
    if (vkCreateInstance(&createInfo, nullptr, &_instance) != VK_SUCCESS)
        throw std::runtime_error("Failed to create vulkan instance");
}

void Device::createDebugMessenger()
{
    // Create debug messenger with everything except info
    VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
    createInfo.pUserData = nullptr; // optional

    if (CreateDebugUtilsMessengerEXT(_instance, &createInfo, nullptr, &_debugMessenger) != VK_SUCCESS)
        throw std::runtime_error("Failed to create vulkan instance");
}

void Device::createSurface(GLFWwindow* window)
{
    if (glfwCreateWindowSurface(_instance, window, nullptr, &_surface) != VK_SUCCESS)
        throw std::runtime_error("Failed to create window surface");
}

void Device::selectPhysicalDevice()
{
    // Find physical devices
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(_instance, &deviceCount, nullptr);
    if (deviceCount == 0)
        throw std::runtime_error("Failed to find GPUs with vulkan support");
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(_instance, &deviceCount, devices.data());

    // Select a suitable one
    _physicalDevice = VK_NULL_HANDLE;
    for (const auto& device : devices) {
        if (isDeviceSuitable(device)) {
            _physicalDevice = device;
            break;
        }
    }

    if (_physicalDevice == VK_NULL_HANDLE)
        throw std::runtime_error("Failed to find a suitable GPU");
}

void Device::createLogicalDevice()
{
    uint32_t graphicsFamily = _queueFamilies.graphicsFamily.value();
    uint32_t presentFamily = _queueFamilies.presentFamily.value();

    // Set up queue info, concatenating duplicate families
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {graphicsFamily, presentFamily};
    float queuePriority = 1;
    for (uint32_t family : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = family;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    // Set up features
    VkPhysicalDeviceFeatures deviceFeatures = {};

    // Setup the logical device
    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = queueCreateInfos.size();
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();

    // Create the logical device
    if (vkCreateDevice(_physicalDevice, &createInfo, nullptr, &_device) != VK_SUCCESS)
        throw std::runtime_error("Failed to create logical device");

    // Get the created queue
    vkGetDeviceQueue(_device, graphicsFamily, 0, &_graphicsQueue);
    vkGetDeviceQueue(_device, presentFamily, 0, &_presentQueue);
}

void Device::createCommandPool()
{
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = _queueFamilies.graphicsFamily.value();
    poolInfo.flags = 0; // optional

    if(vkCreateCommandPool(_device, &poolInfo, nullptr, &_commandPool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create command pool");
}
