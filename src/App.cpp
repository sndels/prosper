#include "App.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {
    const uint32_t WIDTH = 1280;
    const uint32_t HEIGHT = 720;
    const size_t MAX_FRAMES_IN_FLIGHT = 2;
    const std::vector<const char*> validationLayers = {
        //"VK_LAYER_LUNARG_api_dump",
        "VK_LAYER_LUNARG_standard_validation"
    };
    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    static std::vector<char> readFile(const std::string& filename)
    {
        // Open from end to find size from initial position
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open())
            throw  std::runtime_error(std::string("Failed to open file '") + filename + "'");

        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);

        // Seek to beginning and read
        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();
        return buffer;
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface)
    {
        QueueFamilyIndices indices;

        // Get supported queue families
        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, families.data());

        // Find needed queue support
        for (uint32_t i = 0; i < familyCount; ++i) {
            if (families[i].queueCount > 0) {
                // Query present support
                VkBool32 presentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

                // Set to matching families 
                if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                    indices.graphicsFamily = i;
                if (presentSupport)
                    indices.presentFamily = i;
            }

            if (indices.isComplete())
                break;
        }

        return indices;
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface)
    {
        SwapChainSupportDetails details;

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

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
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

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
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

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
    {
        // Check if we have a fixed extent
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
            return capabilities.currentExtent;

        // Pick best resolution from given bounds
        VkExtent2D actualExtent = {};
        actualExtent.width = std::clamp(WIDTH, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(HEIGHT, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device)
    {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

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
}

void App::run() 
{
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

void App::initWindow()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    _window = glfwCreateWindow(WIDTH, HEIGHT, "prosper", nullptr, nullptr);
}

void App::initVulkan()
{
    createInstance();
    createDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandPool();
    createCommandBuffers();
    createSyncObjects();
}

void App::createInstance()
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
    if (vkCreateInstance(&createInfo, nullptr, &_vkInstance) != VK_SUCCESS)
        throw std::runtime_error("Failed to create vulkan instance");
}

void App::createDebugMessenger()
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

    if (CreateDebugUtilsMessengerEXT(_vkInstance, &createInfo, nullptr, &_debugMessenger) != VK_SUCCESS)
        throw std::runtime_error("Failed to create vulkan instance");
}

void App::createSurface()
{
    if (glfwCreateWindowSurface(_vkInstance, _window, nullptr, &_vkSurface) != VK_SUCCESS)
        throw std::runtime_error("Failed to create window surface");
}

bool App::isDeviceSuitable(VkPhysicalDevice device)
{
    QueueFamilyIndices indices = findQueueFamilies(device, _vkSurface);

    bool extensionsSupported = checkDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
        SwapChainSupportDetails swapDetails = querySwapChainSupport(device, _vkSurface);
        swapChainAdequate = !swapDetails.formats.empty() && !swapDetails.presentModes.empty();
    }

    return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

void App::pickPhysicalDevice()
{
    // Find physical devices
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(_vkInstance, &deviceCount, nullptr);

    if (deviceCount == 0)
        throw std::runtime_error("Failed to find GPUs with vulkan support");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(_vkInstance, &deviceCount, devices.data());

    // Pick a suitable one
    _vkPhysicalDevice = VK_NULL_HANDLE;
    for (const auto& device : devices) {
        // Simply check that all needed queues are supported
        if (isDeviceSuitable(device)) {
            _vkPhysicalDevice = device;
            break;
        }
    }

    if (_vkPhysicalDevice == VK_NULL_HANDLE)
        throw std::runtime_error("Failed to find a suitable GPU");
}

void App::createLogicalDevice()
{
    QueueFamilyIndices indices = findQueueFamilies(_vkPhysicalDevice, _vkSurface);

    // Set up queue info, concatenating duplicate families
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};
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
    if (vkCreateDevice(_vkPhysicalDevice, &createInfo, nullptr, &_vkDevice) != VK_SUCCESS)
        throw std::runtime_error("Failed to create logical device");

    // Get the created queue
    vkGetDeviceQueue(_vkDevice, indices.graphicsFamily.value(), 0, &_graphicsQueue);
    vkGetDeviceQueue(_vkDevice, indices.presentFamily.value(), 0, &_presentQueue);
}

void App::createSwapChain()
{
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(_vkPhysicalDevice, _vkSurface);

    // Prepare data
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    // Prefer one extra image to limit waiting on internal operations
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        imageCount = swapChainSupport.capabilities.maxImageCount;

    QueueFamilyIndices indices = findQueueFamilies(_vkPhysicalDevice, _vkSurface);
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

    // Fill out info
    VkSwapchainCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = _vkSurface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
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

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform; // Do mirrors, flips here
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // Opaque window
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE; // Don't care about pixels covered by other windows
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(_vkDevice, &createInfo, nullptr, &_vkSwapchain) != VK_SUCCESS)
        throw std::runtime_error("Failed to create swap chain");

    vkGetSwapchainImagesKHR(_vkDevice, _vkSwapchain, &imageCount, nullptr);
    _vkSwapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(_vkDevice, _vkSwapchain, &imageCount, _vkSwapchainImages.data());
    _vkSwapchainImageFormat = surfaceFormat.format;
    _vkSwapchainExtent = extent;
}

void App::createImageViews()
{
    // Create simple image views to treat swap chain images as color targets
    _vkSwapchainImageViews.resize(_vkSwapchainImages.size());
    for (size_t i = 0; i < _vkSwapchainImages.size(); ++i) {
        VkImageViewCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = _vkSwapchainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = _vkSwapchainImageFormat;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(_vkDevice, &createInfo, nullptr, &_vkSwapchainImageViews[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create image views");
    }
}

void App::createRenderPass()
{
    // Fill color attachment data for the swap buffer
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = _vkSwapchainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // Fill reference to it
    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // Create subpass for output
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    // Create subpass dependency from last pass to synchronize render passess
    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask =  0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    // Create render pass
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(_vkDevice, &renderPassInfo, nullptr, &_vkRenderPass) != VK_SUCCESS)
        throw std::runtime_error("Failed to create render pass");
}

void App::createGraphicsPipeline()
{
    // Create modules for shaders
    auto vertSPV = readFile("shader/shader.vert.spv");
    auto fragSPV = readFile("shader/shader.frag.spv");
    VkShaderModule vertShaderModule = createShaderModule(vertSPV);
    VkShaderModule fragShaderModule = createShaderModule(fragSPV);

    // Fill out create infos for the shader stages
    VkPipelineShaderStageCreateInfo vertStageInfo = {};
    vertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStageInfo.module = vertShaderModule;
    vertStageInfo.pName = "main";
    vertStageInfo.pSpecializationInfo = nullptr; // optional, can set shader constants

    VkPipelineShaderStageCreateInfo fragStageInfo = {};
    fragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStageInfo.module = fragShaderModule;
    fragStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertStageInfo, fragStageInfo};

    // Fill out shader stage inputs
    VkPipelineVertexInputStateCreateInfo vertInputInfo = {};
    vertInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertInputInfo.vertexBindingDescriptionCount = 0;
    vertInputInfo.pVertexBindingDescriptions = nullptr;
    vertInputInfo.vertexAttributeDescriptionCount = 0;
    vertInputInfo.pVertexAttributeDescriptions = nullptr;

    // Fill out input topology
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // Set up viewport
    VkViewport viewport = {};
    viewport.x = 0.f;
    viewport.y = 0.f;
    viewport.width = (float) _vkSwapchainExtent.width;
    viewport.height = (float) _vkSwapchainExtent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = _vkSwapchainExtent;

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    // Fill out rasterizer config
    VkPipelineRasterizationStateCreateInfo rasterizerState = {};
    rasterizerState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizerState.depthClampEnable = VK_FALSE;
    rasterizerState.rasterizerDiscardEnable = VK_FALSE;
    rasterizerState.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizerState.lineWidth = 1.f;
    rasterizerState.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizerState.frontFace = VK_FRONT_FACE_CLOCKWISE; // Clockwise in _screenspace_ with y down
    rasterizerState.depthBiasEnable = VK_FALSE;
    rasterizerState.depthBiasConstantFactor = 0.f; // optional
    rasterizerState.depthBiasClamp = 0.f; // optional
    rasterizerState.depthBiasSlopeFactor = 0.f; // optional

    // Fill out multisampling config
    VkPipelineMultisampleStateCreateInfo multisampleState = {};
    multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampleState.sampleShadingEnable = VK_FALSE;
    multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampleState.minSampleShading = 1.f; // optional
    multisampleState.pSampleMask = nullptr; // optional
    multisampleState.alphaToCoverageEnable = VK_FALSE; // optional
    multisampleState.alphaToOneEnable = VK_FALSE; // optional

    // Fill out blending config
    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // optional
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // optional
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // optional
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // optional
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // optional
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // optional

    VkPipelineColorBlendStateCreateInfo colorBlendState = {};
    colorBlendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendState.logicOpEnable = VK_FALSE;
    colorBlendState.logicOp = VK_LOGIC_OP_COPY; // optional
    colorBlendState.attachmentCount = 1;
    colorBlendState.pAttachments = &colorBlendAttachment;
    colorBlendState.blendConstants[0] = 0.f; //optional
    colorBlendState.blendConstants[1] = 0.f; //optional
    colorBlendState.blendConstants[2] = 0.f; //optional
    colorBlendState.blendConstants[3] = 0.f; //optional

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0; // optional
    pipelineLayoutInfo.pSetLayouts = nullptr; // optional
    pipelineLayoutInfo.pushConstantRangeCount = 0; // optional
    pipelineLayoutInfo.pPushConstantRanges = nullptr; // optional

    if (vkCreatePipelineLayout(_vkDevice, &pipelineLayoutInfo, nullptr, &_vkGraphicsPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create pipeline layout");

    // Create pipeline
    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizerState;
    pipelineInfo.pMultisampleState = &multisampleState;
    pipelineInfo.pDepthStencilState = nullptr; // optional
    pipelineInfo.pColorBlendState = &colorBlendState;
    pipelineInfo.pDynamicState = nullptr; // optional
    pipelineInfo.layout = _vkGraphicsPipelineLayout;
    pipelineInfo.renderPass = _vkRenderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // optional
    pipelineInfo.basePipelineIndex = -1; // optional

    if (vkCreateGraphicsPipelines(_vkDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &_vkGraphicsPipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to crate graphics pipeline");

    vkDestroyShaderModule(_vkDevice, vertShaderModule, nullptr);
    vkDestroyShaderModule(_vkDevice, fragShaderModule, nullptr);
}

VkShaderModule App::createShaderModule(const std::vector<char>& spv)
{
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spv.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(spv.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(_vkDevice, &createInfo, nullptr, &shaderModule))
        throw std::runtime_error("Failed to create shader module");

    return shaderModule;
}

void App::createFramebuffers()
{
    _vkSwapchainFramebuffers.resize(_vkSwapchainImageViews.size());

    for (size_t i = 0; i < _vkSwapchainImageViews.size(); ++i) {
        VkImageView attachments[] = {
            _vkSwapchainImageViews[i]
        };

        VkFramebufferCreateInfo  framebufferInfo = {};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = _vkRenderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = _vkSwapchainExtent.width;
        framebufferInfo.height = _vkSwapchainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(_vkDevice, &framebufferInfo, nullptr, &_vkSwapchainFramebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create framebuffer");
    }
}

void App::createCommandPool()
{
    QueueFamilyIndices indices = findQueueFamilies(_vkPhysicalDevice, _vkSurface);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = indices.graphicsFamily.value();
    poolInfo.flags = 0; // optional

    if(vkCreateCommandPool(_vkDevice, &poolInfo, nullptr, &_vkCommandPool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create command pool");
}

void App::createCommandBuffers()
{
    _vkCommandBuffers.resize(_vkSwapchainFramebuffers.size());

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = _vkCommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t) _vkCommandBuffers.size();

    if (vkAllocateCommandBuffers(_vkDevice, &allocInfo, _vkCommandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate command buffers");

    for (size_t i = 0; i < _vkCommandBuffers.size(); ++i) {
        // Begin command buffer
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT; // Might be scheduling next frame while last is rendering
        beginInfo.pInheritanceInfo = nullptr;

        if (vkBeginCommandBuffer(_vkCommandBuffers[i], &beginInfo) != VK_SUCCESS)
            throw std::runtime_error("Failed to begin recording command buffer");

        // Begin renderpass
        VkClearValue clearColor = {0.f, 0.f, 0.f, 0.f};
        VkRenderPassBeginInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = _vkRenderPass;
        renderPassInfo.framebuffer = _vkSwapchainFramebuffers[i];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = _vkSwapchainExtent;
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(_vkCommandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(_vkCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, _vkGraphicsPipeline);

        vkCmdDraw(_vkCommandBuffers[i], 3, 1, 0, 0);

        vkCmdEndRenderPass(_vkCommandBuffers[i]);

        if (vkEndCommandBuffer(_vkCommandBuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to record command buffer");
    }
}

void App::createSyncObjects()
{
    _imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    _renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    _inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (vkCreateSemaphore(_vkDevice, &semaphoreInfo, nullptr, &_imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(_vkDevice, &semaphoreInfo, nullptr, &_renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(_vkDevice, &fenceInfo, nullptr, &_inFlightFences[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create semaphores");
    }
}

void App::drawFrame()
{
    vkWaitForFences(_vkDevice, 1, &_inFlightFences[_currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());
    vkResetFences(_vkDevice, 1, &_inFlightFences[_currentFrame]);

    uint32_t imageIndex;
    vkAcquireNextImageKHR(_vkDevice, _vkSwapchain, std::numeric_limits<uint64_t>::max(), _imageAvailableSemaphores[_currentFrame], VK_NULL_HANDLE, &imageIndex);

    // Submit queue
    VkSemaphore waitSemaphores[] = {_imageAvailableSemaphores[_currentFrame]};
    VkSemaphore signalSemaphores[] = {_renderFinishedSemaphores[_currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &_vkCommandBuffers[imageIndex];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(_graphicsQueue, 1, &submitInfo, _inFlightFences[_currentFrame]) != VK_SUCCESS)
        throw std::runtime_error("Failed to submit draw command buffer");

    // Present
    VkSwapchainKHR swapchains[] = {_vkSwapchain};
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr; // optional

    vkQueuePresentKHR(_presentQueue, &presentInfo);

    _currentFrame = (_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void App::mainLoop()
{
    while (!glfwWindowShouldClose(_window)) {
        glfwPollEvents();
        drawFrame();
    }

    // Wait for in flight rendering actions to finish
    vkDeviceWaitIdle(_vkDevice);
}

void App::cleanup()
{
    // Destroy vulkan stuff
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkDestroySemaphore(_vkDevice, _renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(_vkDevice, _imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(_vkDevice, _inFlightFences[i], nullptr);
    }
    vkDestroyCommandPool(_vkDevice, _vkCommandPool, nullptr); // Also cleans up associated command buffers
    for (auto framebuffer : _vkSwapchainFramebuffers)
        vkDestroyFramebuffer(_vkDevice, framebuffer, nullptr);
    vkDestroyPipeline(_vkDevice, _vkGraphicsPipeline, nullptr);
    vkDestroyPipelineLayout(_vkDevice, _vkGraphicsPipelineLayout, nullptr);
    vkDestroyRenderPass(_vkDevice, _vkRenderPass, nullptr);
    for (auto imageView : _vkSwapchainImageViews)
        vkDestroyImageView(_vkDevice, imageView, nullptr);
    vkDestroySwapchainKHR(_vkDevice, _vkSwapchain, nullptr);
    vkDestroyDevice(_vkDevice, nullptr); // Implicitly cleans up associated queues as well
    DestroyDebugUtilsMessengerEXT(_vkInstance, _debugMessenger, nullptr);
    vkDestroySurfaceKHR(_vkInstance, _vkSurface, nullptr);
    vkDestroyInstance(_vkInstance, nullptr);

    // Destroy glfw
    glfwDestroyWindow(_window);
    glfwTerminate();
}
