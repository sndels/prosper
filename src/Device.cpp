#include "Device.hpp"

#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>

#include "App.hpp"
#include "Swapchain.hpp"
#include "Utils.hpp"
#include "VkUtils.hpp"

namespace
{

const std::vector<const char *> validationLayers = {
    //"VK_LAYER_LUNARG_api_dump",
    "VK_LAYER_KHRONOS_validation"};
const std::vector<const char *> deviceExtensions = {
    VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
};

QueueFamilies findQueueFamilies(
    const vk::PhysicalDevice device, const vk::SurfaceKHR surface)
{
    const auto allFamilies = device.getQueueFamilyProperties();

    // Find needed queue support
    QueueFamilies families;
    for (uint32_t i = 0; i < allFamilies.size(); ++i)
    {
        if (allFamilies[i].queueCount > 0)
        {
            const vk::Bool32 presentSupport =
                device.getSurfaceSupportKHR(i, surface);

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
    const auto availableExtensions =
        device.enumerateDeviceExtensionProperties(nullptr);

    // Check that all needed extensions are present
    std::set<std::string> requiredExtensions(
        deviceExtensions.begin(), deviceExtensions.end());
    for (const auto &extension : availableExtensions)
    {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

bool checkValidationLayerSupport()
{
    const auto availableLayers = vk::enumerateInstanceLayerProperties();

    // Check that each layer is supported
    for (const char *layerName : validationLayers)
    {
        bool layerFound = false;

        for (const auto &layerProperties : availableLayers)
        {
            if (strcmp(layerName, layerProperties.layerName) == 0)
            {
                layerFound = true;
                break;
            }
        }

        if (!layerFound)
        {
            return false;
        }
    }

    return true;
}

std::vector<const char *> getRequiredExtensions()
{
    // Query extensions glfw requires
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions =
        glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char *> extensions(
        glfwExtensions, glfwExtensions + glfwExtensionCount);

    // Add extension containing debug layers
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    return extensions;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    const VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    const VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData)
{
    (void)messageSeverity;
    (void)messageType;
    (void)pUserData;

    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE; // Don't fail the causing command
}

void CreateDebugUtilsMessengerEXT(
    const vk::Instance instance,
    const vk::DebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const vk::AllocationCallbacks *pAllocator,
    vk::DebugUtilsMessengerEXT *pDebugMessenger)
{
    auto vkInstance = static_cast<VkInstance>(instance);
    auto vkpCreateInfo =
        reinterpret_cast<const VkDebugUtilsMessengerCreateInfoEXT *>(
            pCreateInfo);
    auto vkpAllocator =
        reinterpret_cast<const VkAllocationCallbacks *>(pAllocator);
    auto vkpDebugMessenger =
        reinterpret_cast<VkDebugUtilsMessengerEXT *>(pDebugMessenger);

    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        vkInstance, "vkCreateDebugUtilsMessengerEXT");
    if (func == nullptr ||
        func(vkInstance, vkpCreateInfo, vkpAllocator, vkpDebugMessenger) !=
            VK_SUCCESS)
        throw std::runtime_error("failed to create debug messenger");
}

void DestroyDebugUtilsMessengerEXT(
    const vk::Instance instance,
    const vk::DebugUtilsMessengerEXT debugMessenger,
    const vk::AllocationCallbacks *pAllocator)
{
    auto vkInstance = static_cast<VkInstance>(instance);
    const auto vkDebugMessenger =
        static_cast<const VkDebugUtilsMessengerEXT>(debugMessenger);
    auto vkpAllocator =
        reinterpret_cast<const VkAllocationCallbacks *>(pAllocator);

    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        vkInstance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr)
        func(vkInstance, vkDebugMessenger, vkpAllocator);
}

const char *statusString(shaderc_compilation_status status)
{
    switch (status)
    {
    case shaderc_compilation_status_success:
        return "Success";
    case shaderc_compilation_status_invalid_stage:
        return "Stage deduction failed";
    case shaderc_compilation_status_compilation_error:
        return "Compilation error";
    case shaderc_compilation_status_internal_error:
        return "Internal error";
    case shaderc_compilation_status_null_result_object:
        return "Null result object";
    case shaderc_compilation_status_invalid_assembly:
        return "Invalid assembly";
    case shaderc_compilation_status_validation_error:
        return "Validation error";
    case shaderc_compilation_status_transformation_error:
        return "Transformation error";
    case shaderc_compilation_status_configuration_error:
        return "Configuration error";
    default:
        throw std::runtime_error("Unknown shaderc compilationstatus");
    }
}

} // namespace

vk::BufferMemoryBarrier2KHR TexelBuffer::transitionBarrier(
    const BufferState &newState)
{
    const vk::BufferMemoryBarrier2KHR barrier{
        .srcStageMask = state.stageMask,
        .srcAccessMask = state.accessMask,
        .dstStageMask = newState.stageMask,
        .dstAccessMask = newState.accessMask,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = handle,
        .offset = 0,
        .size = size,
    };

    state = newState;

    return barrier;
}

void TexelBuffer::transition(
    const vk::CommandBuffer buffer, const BufferState &newState)
{
    auto barrier = transitionBarrier(newState);
    buffer.pipelineBarrier2KHR(vk::DependencyInfoKHR{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &barrier,
    });
}

vk::ImageMemoryBarrier2KHR Image::transitionBarrier(const ImageState &newState)
{
    const vk::ImageMemoryBarrier2KHR barrier{
        .srcStageMask = state.stageMask,
        .srcAccessMask = state.accessMask,
        .dstStageMask = newState.stageMask,
        .dstAccessMask = newState.accessMask,
        .oldLayout = state.layout,
        .newLayout = newState.layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = handle,
        .subresourceRange = subresourceRange};

    state = newState;

    return barrier;
}

void Image::transition(
    const vk::CommandBuffer buffer, const ImageState &newState)
{
    auto barrier = transitionBarrier(newState);
    buffer.pipelineBarrier2KHR(vk::DependencyInfoKHR{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    });
}

FileIncluder::FileIncluder()
: _includePath{resPath("shader")}
{
}

shaderc_include_result *FileIncluder::GetInclude(
    const char *requested_source, shaderc_include_type type,
    const char * /*requesting_source*/, size_t /*include_depth*/)
{
    assert(type == shaderc_include_type_relative);

    const auto source = readFileString(_includePath / requested_source);

    char *content = new char[source.size()];
    memcpy((void *)content, source.c_str(), source.size());

    auto *result = new shaderc_include_result;

    result->source_name = requested_source;
    result->source_name_length = strlen(requested_source);
    result->content = content;
    result->content_length = source.size();
    result->user_data = content;

    return result;
}

void FileIncluder::ReleaseInclude(shaderc_include_result *data)
{
    delete[] data->user_data;
    delete data;
}

Device::Device(GLFWwindow *window)
{
    _compilerOptions.SetIncluder(std::make_unique<FileIncluder>());

    vk::DynamicLoader dl;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr =
        dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    createInstance();
    VULKAN_HPP_DEFAULT_DISPATCHER.init(_instance);

    createDebugMessenger();
    createSurface(window);
    selectPhysicalDevice();
    _queueFamilies = findQueueFamilies(_physical, _surface);

    createLogicalDevice();
    VULKAN_HPP_DEFAULT_DISPATCHER.init(_logical);

    createAllocator();
    createCommandPools();

    {
        const auto properties = _physical.getProperties();

        const auto apiPacked = properties.apiVersion;
        fprintf(
            stderr, "Vulkan %u.%u.%u\n", VK_API_VERSION_MAJOR(apiPacked),
            VK_API_VERSION_MINOR(apiPacked), VK_API_VERSION_PATCH(apiPacked));

        fprintf(stderr, "%s\n", properties.deviceName.data());
        fprintf(
            stderr, "Max per descriptor set samplers: %u\n",
            properties.limits.maxDescriptorSetSamplers);
        fprintf(
            stderr, "Max per stage samplers: %u\n",
            properties.limits.maxPerStageDescriptorSamplers);
        fprintf(
            stderr, "Max per descriptor set sampled images: %u\n",
            properties.limits.maxDescriptorSetSampledImages);
        fprintf(
            stderr, "Max per stage sampled images: %u\n",
            properties.limits.maxPerStageDescriptorSampledImages);
    }
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

vk::Instance Device::instance() const { return _instance; }

vk::PhysicalDevice Device::physical() const { return _physical; }

vk::Device Device::logical() const { return _logical; }

vk::SurfaceKHR Device::surface() const { return _surface; }

vk::CommandPool Device::computePool() const { return _graphicsPool; }

vk::CommandPool Device::graphicsPool() const { return _graphicsPool; }

vk::Queue Device::computeQueue() const { return _computeQueue; }

vk::Queue Device::graphicsQueue() const { return _graphicsQueue; }

vk::Queue Device::presentQueue() const { return _presentQueue; }

const QueueFamilies &Device::queueFamilies() const { return _queueFamilies; }

std::optional<vk::ShaderModule> Device::compileShaderModule(
    const std::string &relPath, const std::string &debugName) const
{
    return compileShaderModule(
        readFileString(resPath(relPath)), relPath, debugName);
}

std::optional<vk::ShaderModule> Device::compileShaderModule(
    const std::string &source, const std::string &path,
    const std::string &debugName) const
{
    const auto result = _compiler.CompileGlslToSpv(
        source, shaderc_glsl_infer_from_source, path.c_str(), _compilerOptions);

    if (const auto status = result.GetCompilationStatus(); status)
    {
        const auto err = result.GetErrorMessage();
        if (!err.empty())
            fprintf(stderr, "%s\n", err.c_str());
        fprintf(stderr, "Compilation of '%s' failed\n", path.c_str());
        fprintf(stderr, "%s\n", statusString(status));
        return {};
    }

    const auto sm = _logical.createShaderModule(vk::ShaderModuleCreateInfo{
        .codeSize = static_cast<size_t>(result.end() - result.begin()) *
                    sizeof(uint32_t),
        .pCode = result.begin(),
    });

    _logical.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
        .objectType = vk::ObjectType::eShaderModule,
        .objectHandle =
            reinterpret_cast<uint64_t>(static_cast<VkShaderModule>(sm)),
        .pObjectName = debugName.c_str(),
    });

    return sm;
}

void Device::map(const VmaAllocation allocation, void **data) const
{
    vmaMapMemory(_allocator, allocation, data);
}

void Device::unmap(const VmaAllocation allocation) const
{
    vmaUnmapMemory(_allocator, allocation);
}

Buffer Device::createBuffer(
    const std::string &debugName, const vk::DeviceSize size,
    const vk::BufferUsageFlags usage, const vk::MemoryPropertyFlags properties,
    const VmaMemoryUsage vmaUsage) const
{
    vk::BufferCreateInfo bufferInfo{
        .size = size,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive};
    // TODO: preferred flags, create mapped
    VmaAllocationCreateInfo allocInfo = {
        .usage = vmaUsage,
        .requiredFlags = static_cast<VkMemoryPropertyFlags>(properties),
    };

    Buffer buffer;
    auto vkpBufferInfo = reinterpret_cast<VkBufferCreateInfo *>(&bufferInfo);
    auto vkpBuffer = reinterpret_cast<VkBuffer *>(&buffer.handle);
    vmaCreateBuffer(
        _allocator, vkpBufferInfo, &allocInfo, vkpBuffer, &buffer.allocation,
        nullptr);

    _logical.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
        .objectType = vk::ObjectType::eBuffer,
        .objectHandle =
            reinterpret_cast<uint64_t>(static_cast<VkBuffer>(buffer.handle)),
        .pObjectName = debugName.c_str()});

    return buffer;
}

void Device::destroy(const Buffer &buffer) const
{
    const auto vkBuffer = static_cast<VkBuffer>(buffer.handle);
    vmaDestroyBuffer(_allocator, vkBuffer, buffer.allocation);
}

TexelBuffer Device::createTexelBuffer(
    const std::string &debugName, const vk::Format format,
    const vk::DeviceSize size, const vk::BufferUsageFlags usage,
    const vk::MemoryPropertyFlags properties, const bool supportAtomics,
    const VmaMemoryUsage vmaUsage) const
{
    const auto formatProperties = _physical.getFormatProperties(format);

    if (containsFlag(usage, vk::BufferUsageFlagBits::eStorageTexelBuffer))
    {
        assertContainsFlag(
            formatProperties.bufferFeatures,
            vk::FormatFeatureFlagBits::eStorageTexelBuffer,
            "Format doesn't support storage texel buffer");
    }
    if (containsFlag(usage, vk::BufferUsageFlagBits::eUniformTexelBuffer))
    {
        assertContainsFlag(
            formatProperties.bufferFeatures,
            vk::FormatFeatureFlagBits::eUniformTexelBuffer,
            "Format doesn't support uniform texel buffer");
    }
    if (supportAtomics)
    {
        assertContainsFlag(
            formatProperties.bufferFeatures,
            vk::FormatFeatureFlagBits::eStorageTexelBufferAtomic,
            "Format doesn't support atomics");
    }

    const auto [handle, allocation] =
        createBuffer(debugName, size, usage, properties, vmaUsage);

    const auto view = _logical.createBufferView(vk::BufferViewCreateInfo{
        .buffer = handle,
        .format = format,
        .offset = 0,
        .range = size,
    });

    return TexelBuffer{
        .handle = handle,
        .view = view,
        .format = format,
        .size = size,
        .allocation = allocation,
    };
}

void Device::destroy(const TexelBuffer &buffer) const
{
    const auto vkBuffer = static_cast<VkBuffer>(buffer.handle);
    vmaDestroyBuffer(_allocator, vkBuffer, buffer.allocation);
    _logical.destroy(buffer.view);
}

Image Device::createImage(
    const std::string &debugName, const vk::Extent2D extent,
    const vk::Format format, const vk::ImageSubresourceRange &range,
    const vk::ImageViewType viewType, const vk::ImageTiling tiling,
    const vk::ImageCreateFlags flags, const vk::ImageUsageFlags usage,
    const vk::MemoryPropertyFlags properties,
    const VmaMemoryUsage vmaUsage) const
{
    vk::ImageCreateInfo imageInfo{
        .flags = flags,
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = vk::Extent3D{extent.width, extent.height, 1},
        .mipLevels = range.levelCount,
        .arrayLayers = range.layerCount,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = tiling,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive};
    // TODO: preferred flags, create mapped
    VmaAllocationCreateInfo allocInfo = {
        .usage = vmaUsage,
        .requiredFlags = static_cast<VkMemoryPropertyFlags>(properties)};

    Image image;
    auto vkpImageInfo = reinterpret_cast<VkImageCreateInfo *>(&imageInfo);
    auto vkpImage = reinterpret_cast<VkImage *>(&image.handle);
    vmaCreateImage(
        _allocator, vkpImageInfo, &allocInfo, vkpImage, &image.allocation,
        nullptr);

    _logical.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
        .objectType = vk::ObjectType::eImage,
        .objectHandle =
            reinterpret_cast<uint64_t>(static_cast<VkImage>(image.handle)),
        .pObjectName = debugName.c_str()});

    image.view = _logical.createImageView(vk::ImageViewCreateInfo{
        .image = image.handle,
        .viewType = viewType,
        .format = format,
        .subresourceRange = range});

    image.extent = extent;
    image.subresourceRange = range;
    image.format = format;
    return image;
}

void Device::destroy(const Image &image) const
{
    const auto vkImage = static_cast<VkImage>(image.handle);
    vmaDestroyImage(_allocator, vkImage, image.allocation);
    _logical.destroy(image.view);
}

vk::CommandBuffer Device::beginGraphicsCommands() const
{
    const auto buffer =
        _logical.allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = _graphicsPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1})[0];

    const vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
    buffer.begin(beginInfo);

    return buffer;
}

void Device::endGraphicsCommands(const vk::CommandBuffer buffer) const
{
    buffer.end();

    const vk::SubmitInfo submitInfo{
        .commandBufferCount = 1, .pCommandBuffers = &buffer};
    checkSuccess(_graphicsQueue.submit(1, &submitInfo, vk::Fence{}), "submit");
    _graphicsQueue.waitIdle();

    _logical.freeCommandBuffers(_graphicsPool, 1, &buffer);
}

bool Device::isDeviceSuitable(const vk::PhysicalDevice device) const
{
    const auto families = findQueueFamilies(device, _surface);

    const auto extensionsSupported = checkDeviceExtensionSupport(device);

    vk::PhysicalDeviceDynamicRenderingFeaturesKHR dynamicRenderingFeatures{};
    vk::PhysicalDeviceSynchronization2FeaturesKHR sync2Features{
        .pNext = &dynamicRenderingFeatures,
    };
    vk::PhysicalDeviceVulkan12Features vk12Features{
        .pNext = &sync2Features,
        .descriptorIndexing = VK_TRUE,
        .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
        .runtimeDescriptorArray = VK_TRUE,
    };
    vk::PhysicalDeviceFeatures2 supportedFeatures{
        .pNext = &vk12Features,
        .features =
            {
                .samplerAnisotropy = VK_TRUE,
                .shaderSampledImageArrayDynamicIndexing = VK_TRUE,
            },
    };
    device.getFeatures2(&supportedFeatures);

    const bool swapChainAdequate = [&]
    {
        bool adequate = false;
        if (extensionsSupported)
        {
            SwapchainSupport swapSupport{device, _surface};
            adequate = !swapSupport.formats.empty() &&
                       !swapSupport.presentModes.empty();
        }

        return adequate;
    }();

    return families.isComplete() && extensionsSupported && swapChainAdequate &&
           supportedFeatures.features.samplerAnisotropy &&
           sync2Features.synchronization2 &&
           dynamicRenderingFeatures.dynamicRendering;
}

void Device::createInstance()
{
    if (!checkValidationLayerSupport())
        throw std::runtime_error("Validation layers not available");

    const vk::ApplicationInfo appInfo{
        .pApplicationName = "prosper",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "prosper",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_2};

    const auto extensions = getRequiredExtensions();

    _instance = vk::createInstance(vk::InstanceCreateInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(validationLayers.size()),
        .ppEnabledLayerNames = validationLayers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data()});
}

void Device::createDebugMessenger()
{
    const vk::DebugUtilsMessengerCreateInfoEXT createInfo{
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        .pfnUserCallback = debugCallback};
    CreateDebugUtilsMessengerEXT(
        _instance, &createInfo, nullptr, &_debugMessenger);
}

void Device::createSurface(GLFWwindow *window)
{
    auto vkpSurface = reinterpret_cast<VkSurfaceKHR *>(&_surface);
    auto vkInstance = static_cast<VkInstance>(_instance);
    if (glfwCreateWindowSurface(vkInstance, window, nullptr, vkpSurface) !=
        VK_SUCCESS)
        throw std::runtime_error("Failed to create window surface");
}

void Device::selectPhysicalDevice()
{
    const auto devices = _instance.enumeratePhysicalDevices();

    for (const auto &device : devices)
    {
        if (isDeviceSuitable(device))
        {
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
    const std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos = [&]
    {
        std::vector<vk::DeviceQueueCreateInfo> cis;
        const std::set<uint32_t> uniqueQueueFamilies = {
            computeFamily, graphicsFamily, presentFamily};
        for (uint32_t family : uniqueQueueFamilies)
        {
            cis.push_back(vk::DeviceQueueCreateInfo{
                .queueFamilyIndex = family,
                .queueCount = 1,
                .pQueuePriorities = &queuePriority});
        }
        return cis;
    }();

    vk::PhysicalDeviceDynamicRenderingFeaturesKHR dynamicRenderingFeatures{
        .dynamicRendering = true,
    };
    vk::PhysicalDeviceSynchronization2FeaturesKHR sync2Features{
        .pNext = &dynamicRenderingFeatures,
        .synchronization2 = true,
    };
    vk::PhysicalDeviceVulkan12Features vk12Features{
        .pNext = &sync2Features,
        .descriptorIndexing = VK_TRUE,
        .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
        .runtimeDescriptorArray = VK_TRUE,
    };
    const vk::PhysicalDeviceFeatures2 deviceFeatures{
        .pNext = &vk12Features,
        .features =
            {
                .samplerAnisotropy = VK_TRUE,
                .shaderSampledImageArrayDynamicIndexing = VK_TRUE,
            },
    };

    _logical = _physical.createDevice(vk::DeviceCreateInfo{
        .pNext = &deviceFeatures,
        .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledLayerCount = static_cast<uint32_t>(validationLayers.size()),
        .ppEnabledLayerNames = validationLayers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
    });

    // Get the created queues
    _computeQueue = _logical.getQueue(computeFamily, 0);
    _graphicsQueue = _logical.getQueue(graphicsFamily, 0);
    _presentQueue = _logical.getQueue(presentFamily, 0);
}

void Device::createAllocator()
{
    VmaAllocatorCreateInfo allocatorInfo{
        .physicalDevice = static_cast<VkPhysicalDevice>(_physical),
        .device = static_cast<VkDevice>(_logical)};
    if (vmaCreateAllocator(&allocatorInfo, &_allocator) != VK_SUCCESS)
        throw std::runtime_error("Failed to create allocator");
}

void Device::createCommandPools()
{
    {
        const vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = _queueFamilies.graphicsFamily.value()};
        _graphicsPool = _logical.createCommandPool(poolInfo, nullptr);
    }

    {
        const vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = _queueFamilies.computeFamily.value()};
        _computePool = _logical.createCommandPool(poolInfo, nullptr);
    }
}
