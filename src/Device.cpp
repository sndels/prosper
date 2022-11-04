#include "Device.hpp"

#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>

#include "App.hpp"
#include "ForEach.hpp"
#include "Swapchain.hpp"
#include "Utils.hpp"
#include "VkUtils.hpp"

#define ALL_FEATURE_STRUCTS_LIST                                               \
    vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features,           \
        vk::PhysicalDeviceVulkan13Features,                                    \
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR,                    \
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR

#define REQUIRED_FEATURES                                                      \
    vk::PhysicalDeviceFeatures2, features.geometryShader,                      \
        vk::PhysicalDeviceFeatures2, features.samplerAnisotropy,               \
        vk::PhysicalDeviceFeatures2,                                           \
        features.shaderSampledImageArrayDynamicIndexing,                       \
        vk::PhysicalDeviceVulkan12Features, descriptorIndexing,                \
        vk::PhysicalDeviceVulkan12Features,                                    \
        shaderSampledImageArrayNonUniformIndexing,                             \
        vk::PhysicalDeviceVulkan12Features,                                    \
        descriptorBindingVariableDescriptorCount,                              \
        vk::PhysicalDeviceVulkan12Features, runtimeDescriptorArray,            \
        vk::PhysicalDeviceVulkan12Features, hostQueryReset,                    \
        vk::PhysicalDeviceVulkan12Features, bufferDeviceAddress,               \
        vk::PhysicalDeviceVulkan13Features, synchronization2,                  \
        vk::PhysicalDeviceVulkan13Features, dynamicRendering,                  \
        vk::PhysicalDeviceVulkan13Features, maintenance4,                      \
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR,                    \
        accelerationStructure,                                                 \
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR, rayTracingPipeline

namespace
{

constexpr std::array validationLayers = {
    //"VK_LAYER_LUNARG_api_dump",
    "VK_LAYER_KHRONOS_validation",
};
constexpr std::array deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
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
            if (allFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics)
                families.graphicsFamily = i;
            if (presentSupport == VK_TRUE)
                families.presentFamily = i;
        }

        if (families.isComplete())
            break;
    }

    assert(
        (!families.graphicsFamily ||
         (allFamilies[*families.graphicsFamily].timestampValidBits == 64)) &&
        "All bits assumed to be valid for simplicity in profiler");
    assert(
        (!families.presentFamily ||
         (allFamilies[*families.presentFamily].timestampValidBits == 64)) &&
        "All bits assumed to be valid for simplicity in profiler");

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

    if (!requiredExtensions.empty())
    {
        fprintf(stderr, "Missing support for extensions:\n");
        for (const auto &e : requiredExtensions)
            fprintf(stderr, "  %s\n", e.c_str());
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

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    const VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    const VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData)
{
    (void)messageSeverity;
    (void)messageType;
    (void)pUserData;

    // VK_TRUE is reserved
    constexpr auto ret = VK_FALSE;

#define DEVICE_EXTENSION_STR "Device Extension: "
    // Skip extension dump noise
    if (strncmp(
            DEVICE_EXTENSION_STR, pCallbackData->pMessage,
            sizeof(DEVICE_EXTENSION_STR) - 1) == 0)
        return ret;
#undef DEVICE_EXTENSION_STR

    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return ret;
}

void CreateDebugUtilsMessengerEXT(
    const vk::Instance instance,
    const vk::DebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const vk::AllocationCallbacks *pAllocator,
    vk::DebugUtilsMessengerEXT *pDebugMessenger)
{
    auto *vkInstance = static_cast<VkInstance>(instance);
    const auto *vkpCreateInfo =
        reinterpret_cast<const VkDebugUtilsMessengerCreateInfoEXT *>(
            pCreateInfo);
    const auto *vkpAllocator =
        reinterpret_cast<const VkAllocationCallbacks *>(pAllocator);
    auto *vkpDebugMessenger =
        reinterpret_cast<VkDebugUtilsMessengerEXT *>(pDebugMessenger);

    auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(vkInstance, "vkCreateDebugUtilsMessengerEXT"));
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
    auto *vkInstance = static_cast<VkInstance>(instance);
    auto *vkDebugMessenger =
        static_cast<const VkDebugUtilsMessengerEXT>(debugMessenger);
    const auto *vkpAllocator =
        reinterpret_cast<const VkAllocationCallbacks *>(pAllocator);

    auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(vkInstance, "vkDestroyDebugUtilsMessengerEXT"));
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

FileIncluder::FileIncluder()
: _includePath{resPath("shader")}
{
}

shaderc_include_result *FileIncluder::GetInclude(
    const char *requested_source, shaderc_include_type type,
    const char *requesting_source, size_t include_depth)
{
    if (include_depth > 100)
    {
        throw std::runtime_error(
            std::string{
                "Deep shader include recursion with requested source '"} +
            requested_source + "'. Cycle?");
    }

    assert(type == shaderc_include_type_relative);
    (void)type;

    const auto requestingDir =
        std::filesystem::path{requesting_source}.parent_path();
    const auto requestedSource =
        (requestingDir / requested_source).lexically_normal();
    assert(std::filesystem::exists(requestedSource));

    IncludeContent content;
    content.path = std::make_unique<std::string>();
    *content.path = requestedSource.generic_string();

    content.content = std::make_unique<std::string>();
    *content.content = readFileString(requestedSource);

    content.result =
        std::make_unique<shaderc_include_result>(shaderc_include_result{
            .source_name = content.path->c_str(),
            .source_name_length = content.path->size(),
            .content = content.content->c_str(),
            .content_length = content.content->size(),
            .user_data = reinterpret_cast<void *>(_includeContentID), // NOLINT
        });
    auto *result_ptr = content.result.get();

    static_assert(sizeof(_includeContentID) == sizeof(void *));
    _includeContent[_includeContentID++] = std::move(content);

    return result_ptr;
}

void FileIncluder::ReleaseInclude(shaderc_include_result *data)
{
    auto id = reinterpret_cast<uint64_t>(data->user_data);
    _includeContent.erase(id);
}

Device::Device(GLFWwindow *window, bool enableDebugLayers)
{
    printf("Creating Vulkan device\n");

    _compilerOptions.SetIncluder(std::make_unique<FileIncluder>());
    _compilerOptions.SetGenerateDebugInfo();
    _compilerOptions.SetTargetSpirv(shaderc_spirv_version_1_6);

    vk::DynamicLoader dl;
    auto vkGetInstanceProcAddr =
        dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    createInstance(enableDebugLayers);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(_instance);

    createDebugMessenger();
    createSurface(window);
    selectPhysicalDevice();
    _queueFamilies = findQueueFamilies(_physical, _surface);

    createLogicalDevice(enableDebugLayers);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(_logical);

    createAllocator();
    createCommandPools();

    {
        const auto props = _physical.getProperties2<
            vk::PhysicalDeviceProperties2,
            vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
            vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();
        _properties.device =
            props.get<vk::PhysicalDeviceProperties2>().properties;
        _properties.rtPipeline =
            props.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
        _properties.accelerationStructure =
            props.get<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();

        {
            const auto apiPacked = _properties.device.apiVersion;
            const auto major = VK_API_VERSION_MAJOR(apiPacked);
            const auto minor = VK_API_VERSION_MINOR(apiPacked);
            const auto patch = VK_API_VERSION_PATCH(apiPacked);
            printf("Vulkan %u.%u.%u\n", major, minor, patch);

            if (major < 1 || minor < 3)
                throw std::runtime_error("Vulkan 1.3 required");
        }

        printf("%s\n", _properties.device.deviceName.data());
    }
}

Device::~Device()
{
    // Also cleans up associated command buffers
    _logical.destroy(_graphicsPool);
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

vk::CommandPool Device::graphicsPool() const { return _graphicsPool; }

vk::Queue Device::graphicsQueue() const { return _graphicsQueue; }

vk::Queue Device::presentQueue() const { return _presentQueue; }

const QueueFamilies &Device::queueFamilies() const { return _queueFamilies; }

const DeviceProperties &Device::properties() const { return _properties; }

std::optional<vk::ShaderModule> Device::compileShaderModule(
    CompileShaderModuleArgs const &info) const
{
    assert(info.relPath.starts_with("shader/"));
    const auto shaderPath = resPath(info.relPath);

    // Prepend version, defines and reset line offset before the actual source
    const auto source = "#version 460\n" + info.defines + "#line 1\n" +
                        readFileString(shaderPath);

    const auto result = _compiler.CompileGlslToSpv(
        source, shaderc_glsl_infer_from_source, shaderPath.string().c_str(),
        _compilerOptions);

    if (const auto status = result.GetCompilationStatus(); status)
    {
        const auto err = result.GetErrorMessage();
        if (!err.empty())
            fprintf(stderr, "%s\n", err.c_str());
        fprintf(
            stderr, "Compilation of '%s' failed\n",
            shaderPath.string().c_str());
        fprintf(stderr, "%s\n", statusString(status));
        return {};
    }

    const auto sm = _logical.createShaderModule(vk::ShaderModuleCreateInfo{
        .codeSize = asserted_cast<size_t>(result.end() - result.begin()) *
                    sizeof(uint32_t),
        .pCode = result.begin(),
    });

    _logical.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
        .objectType = vk::ObjectType::eShaderModule,
        .objectHandle =
            reinterpret_cast<uint64_t>(static_cast<VkShaderModule>(sm)),
        .pObjectName = info.debugName.c_str(),
    });

    return sm;
}

Buffer Device::createBuffer(const BufferCreateInfo &info)
{
    const vk::BufferCreateInfo bufferInfo{
        .size = info.byteSize,
        .usage = info.usage,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    VmaAllocationCreateFlags allocFlags = 0;
    if ((info.properties & vk::MemoryPropertyFlagBits::eHostVisible) ==
        vk::MemoryPropertyFlagBits::eHostVisible)
        // Readback is not used yet so assume this is for staging
        allocFlags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    if (info.createMapped)
        allocFlags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;

    const VmaAllocationCreateInfo allocCreateInfo = {
        .flags = allocFlags,
        .usage = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags = static_cast<VkMemoryPropertyFlags>(info.properties),
    };

    Buffer buffer;
    VmaAllocationInfo allocInfo;
    const auto *vkpBufferInfo =
        reinterpret_cast<const VkBufferCreateInfo *>(&bufferInfo);
    auto *vkpBuffer = reinterpret_cast<VkBuffer *>(&buffer.handle);
    vmaCreateBuffer(
        _allocator, vkpBufferInfo, &allocCreateInfo, vkpBuffer,
        &buffer.allocation, &allocInfo);

    if (info.createMapped)
    {
        assert(allocInfo.pMappedData);
        buffer.mapped = allocInfo.pMappedData;
    }

    _logical.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
        .objectType = vk::ObjectType::eBuffer,
        .objectHandle =
            reinterpret_cast<uint64_t>(static_cast<VkBuffer>(buffer.handle)),
        .pObjectName = info.debugName.c_str(),
    });

    if (info.initialData != nullptr)
    {
        const auto stagingBuffer = createBuffer(BufferCreateInfo{
            .byteSize = info.byteSize,
            .usage = vk::BufferUsageFlagBits::eTransferSrc,
            .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                          vk::MemoryPropertyFlagBits::eHostCoherent,
            .createMapped = true,
            .debugName = info.debugName + "StagingBuffer",
        });

        memcpy(stagingBuffer.mapped, info.initialData, info.byteSize);

        const auto commandBuffer = beginGraphicsCommands();

        const vk::BufferCopy copyRegion{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = info.byteSize,
        };
        commandBuffer.copyBuffer(
            stagingBuffer.handle, buffer.handle, 1, &copyRegion);

        endGraphicsCommands(commandBuffer);

        destroy(stagingBuffer);
    }

    trackBuffer(buffer);

    return buffer;
}

void Device::destroy(const Buffer &buffer)
{
    untrackBuffer(buffer);

    auto *vkBuffer = static_cast<VkBuffer>(buffer.handle);
    vmaDestroyBuffer(_allocator, vkBuffer, buffer.allocation);
}

TexelBuffer Device::createTexelBuffer(const TexelBufferCreateInfo &info)
{
    const auto formatProperties = _physical.getFormatProperties(info.format);

    if (containsFlag(
            info.bufferInfo.usage,
            vk::BufferUsageFlagBits::eStorageTexelBuffer))
    {
        assertContainsFlag(
            formatProperties.bufferFeatures,
            vk::FormatFeatureFlagBits::eStorageTexelBuffer,
            "Format doesn't support storage texel buffer");
    }
    if (containsFlag(
            info.bufferInfo.usage,
            vk::BufferUsageFlagBits::eUniformTexelBuffer))
    {
        assertContainsFlag(
            formatProperties.bufferFeatures,
            vk::FormatFeatureFlagBits::eUniformTexelBuffer,
            "Format doesn't support uniform texel buffer");
    }
    if (info.supportAtomics)
    {
        assertContainsFlag(
            formatProperties.bufferFeatures,
            vk::FormatFeatureFlagBits::eStorageTexelBufferAtomic,
            "Format doesn't support atomics");
    }

    assert(!info.bufferInfo.createMapped && "Mapped texel buffers not tested");
    assert(
        !info.bufferInfo.initialData &&
        "Texel buffers with initial data not tested");

    const auto buffer = createBuffer(info.bufferInfo);

    const auto view = _logical.createBufferView(vk::BufferViewCreateInfo{
        .buffer = buffer.handle,
        .format = info.format,
        .offset = 0,
        .range = info.bufferInfo.byteSize,
    });

    return TexelBuffer{
        .handle = buffer.handle,
        .view = view,
        .format = info.format,
        .size = info.bufferInfo.byteSize,
        .allocation = buffer.allocation,
    };
}

void Device::destroy(const TexelBuffer &buffer)
{
    untrackTexelBuffer(buffer);

    auto *vkBuffer = static_cast<VkBuffer>(buffer.handle);
    vmaDestroyBuffer(_allocator, vkBuffer, buffer.allocation);
    _logical.destroy(buffer.view);
}

Image Device::createImage(const ImageCreateInfo &info)
{

    const vk::Extent3D extent{
        .width = info.width,
        .height = info.height,
        .depth = info.depth,
    };

    const vk::ImageCreateInfo imageInfo{
        .flags = info.createFlags,
        .imageType = info.imageType,
        .format = info.format,
        .extent = extent,
        .mipLevels = info.mipCount,
        .arrayLayers = info.layerCount,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = info.usageFlags,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const VmaAllocationCreateInfo allocInfo = {
        .flags = {}, // Device only
        .usage = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags = static_cast<VkMemoryPropertyFlags>(info.properties),
    };

    Image image;
    const auto *vkpImageInfo =
        reinterpret_cast<const VkImageCreateInfo *>(&imageInfo);
    auto *vkpImage = reinterpret_cast<VkImage *>(&image.handle);
    vmaCreateImage(
        _allocator, vkpImageInfo, &allocInfo, vkpImage, &image.allocation,
        nullptr);

    _logical.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
        .objectType = vk::ObjectType::eImage,
        .objectHandle =
            reinterpret_cast<uint64_t>(static_cast<VkImage>(image.handle)),
        .pObjectName = info.debugName.c_str(),
    });

    const vk::ImageSubresourceRange range{
        .aspectMask = aspectMask(info.format),
        .baseMipLevel = 0,
        .levelCount = info.mipCount,
        .baseArrayLayer = 0,
        .layerCount = info.layerCount,
    };

    const vk::ImageViewType viewType = [info]()
    {
        switch (info.imageType)
        {
        case vk::ImageType::e1D:
            if (info.layerCount == 1)
                return vk::ImageViewType::e1D;
            else
                return vk::ImageViewType::e1DArray;
        case vk::ImageType::e2D:
            if (info.layerCount == 1)
                return vk::ImageViewType::e2D;
            else
            {
                if ((info.createFlags &
                     vk::ImageCreateFlagBits::eCubeCompatible) ==
                    vk::ImageCreateFlagBits::eCubeCompatible)
                {
                    assert(info.layerCount == 6 && "Cube arrays not supported");
                    return vk::ImageViewType::eCube;
                }
                return vk::ImageViewType::e2DArray;
            }
        case vk::ImageType::e3D:
            assert(info.layerCount == 1 && "Can't have 3D image arrays");
            return vk::ImageViewType::e3D;
        default:
            throw std::runtime_error(
                "Unexpected image type " + to_string(info.imageType));
        }
    }();

    image.view = _logical.createImageView(vk::ImageViewCreateInfo{
        .image = image.handle,
        .viewType = viewType,
        .format = info.format,
        .subresourceRange = range,
    });

    image.extent = extent;
    image.subresourceRange = range;
    image.format = info.format;

    trackImage(image);

    return image;
}

void Device::destroy(const Image &image)
{
    untrackImage(image);

    auto *vkImage = static_cast<VkImage>(image.handle);
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
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    buffer.begin(beginInfo);

    return buffer;
}

void Device::endGraphicsCommands(const vk::CommandBuffer buffer) const
{
    buffer.end();

    const vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &buffer,
    };
    checkSuccess(_graphicsQueue.submit(1, &submitInfo, vk::Fence{}), "submit");
    _graphicsQueue.waitIdle();

    _logical.freeCommandBuffers(_graphicsPool, 1, &buffer);
}

const MemoryAllocationBytes &Device::memoryAllocations() const
{
    return _memoryAllocations;
}

bool Device::isDeviceSuitable(const vk::PhysicalDevice device) const
{
    const auto families = findQueueFamilies(device, _surface);
    if (!families.isComplete())
    {
        fprintf(stderr, "Missing required queue families\n");
        return false;
    }

    if (!checkDeviceExtensionSupport(device))
        return false;

    SwapchainSupport swapSupport{device, _surface};
    if (swapSupport.formats.empty() || swapSupport.presentModes.empty())
    {
        fprintf(stderr, "Inadequate swap chain\n");
        return false;
    }

    printf("Checking feature support\n");

    const auto features = device.getFeatures2<ALL_FEATURE_STRUCTS_LIST>();

#define CHECK_REQUIRED_FEATURES(container, feature)                            \
    if (features.get<container>().feature == VK_FALSE)                         \
    {                                                                          \
        fprintf(stderr, "Missing %s\n", #feature);                             \
        return false;                                                          \
    }

    FOR_EACH_PAIR(CHECK_REQUIRED_FEATURES, REQUIRED_FEATURES);

#undef CHECK_REQUIRED_FEATURES

    printf("Required features are supported\n");

    return true;
}

void *Device::map(VmaAllocation allocation) const
{
    void *mapped = nullptr;
    vmaMapMemory(_allocator, allocation, &mapped);
    return mapped;
}

void Device::unmap(VmaAllocation allocation) const
{
    vmaUnmapMemory(_allocator, allocation);
}

void Device::createInstance(bool enableDebugLayers)
{
    if (enableDebugLayers && !checkValidationLayerSupport())
        throw std::runtime_error("Validation layers not available");

    const vk::ApplicationInfo appInfo{
        .pApplicationName = "prosper",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "prosper",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_3,
    };

    const auto extensions = getRequiredExtensions();

    _instance = vk::createInstance(vk::InstanceCreateInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount =
            enableDebugLayers ? asserted_cast<uint32_t>(validationLayers.size())
                              : 0,
        .ppEnabledLayerNames =
            enableDebugLayers ? validationLayers.data() : nullptr,
        .enabledExtensionCount = asserted_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    });
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
        .pfnUserCallback = debugCallback,
    };
    CreateDebugUtilsMessengerEXT(
        _instance, &createInfo, nullptr, &_debugMessenger);
}

void Device::createSurface(GLFWwindow *window)
{
    auto *vkpSurface = reinterpret_cast<VkSurfaceKHR *>(&_surface);
    auto *vkInstance = static_cast<VkInstance>(_instance);
    if (glfwCreateWindowSurface(vkInstance, window, nullptr, vkpSurface) !=
        VK_SUCCESS)
        throw std::runtime_error("Failed to create window surface");
}

void Device::selectPhysicalDevice()
{
    printf("Selecting device\n");

    const auto devices = _instance.enumeratePhysicalDevices();

    for (const auto &device : devices)
    {
        printf("Considering '%s'\n", device.getProperties().deviceName.data());
        if (isDeviceSuitable(device))
        {
            _physical = device;
            return;
        }
    }

    throw std::runtime_error("Failed to find a suitable GPU");
}

void Device::createLogicalDevice(bool enableDebugLayers)
{
    const uint32_t graphicsFamily = _queueFamilies.graphicsFamily.value();
    const uint32_t presentFamily = _queueFamilies.presentFamily.value();

    // Config queues, concat duplicate families
    const float queuePriority = 1;
    const std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos = [&]
    {
        const std::set<uint32_t> uniqueQueueFamilies = {
            graphicsFamily, presentFamily};

        std::vector<vk::DeviceQueueCreateInfo> cis;
        cis.reserve(uniqueQueueFamilies.size());
        for (auto family : uniqueQueueFamilies)
        {
            cis.push_back(vk::DeviceQueueCreateInfo{
                .queueFamilyIndex = family,
                .queueCount = 1,
                .pQueuePriorities = &queuePriority,
            });
        }

        return cis;
    }();

    vk::StructureChain<vk::DeviceCreateInfo, ALL_FEATURE_STRUCTS_LIST>
        createChain;
    createChain.get<vk::DeviceCreateInfo>() = vk::DeviceCreateInfo{
        .pNext = &createChain.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount =
            asserted_cast<uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledLayerCount =
            enableDebugLayers ? asserted_cast<uint32_t>(validationLayers.size())
                              : 0,
        .ppEnabledLayerNames =
            enableDebugLayers ? validationLayers.data() : nullptr,
        .enabledExtensionCount =
            asserted_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
    };

#define TOGGLE_REQUIRED_FEATURES(container, feature)                           \
    createChain.get<container>().feature = VK_TRUE;

    FOR_EACH_PAIR(TOGGLE_REQUIRED_FEATURES, REQUIRED_FEATURES);

#undef TOGGLE_REQUIRED_FEATURES

    _logical = _physical.createDevice(createChain.get<vk::DeviceCreateInfo>());

    // Get the created queues
    _graphicsQueue = _logical.getQueue(graphicsFamily, 0);
    _presentQueue = _logical.getQueue(presentFamily, 0);
}

void Device::createAllocator()
{
    const VmaAllocatorCreateInfo allocatorInfo{
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = static_cast<VkPhysicalDevice>(_physical),
        .device = static_cast<VkDevice>(_logical),
        .instance = static_cast<VkInstance>(_instance),
    };
    if (vmaCreateAllocator(&allocatorInfo, &_allocator) != VK_SUCCESS)
        throw std::runtime_error("Failed to create allocator");
}

void Device::createCommandPools()
{
    {
        const vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = _queueFamilies.graphicsFamily.value(),
        };
        _graphicsPool = _logical.createCommandPool(poolInfo, nullptr);
    }
}

void Device::trackBuffer(const Buffer &buffer)
{
    VmaAllocationInfo info;
    vmaGetAllocationInfo(_allocator, buffer.allocation, &info);

    _memoryAllocations.buffers += info.size;
}

void Device::untrackBuffer(const Buffer &buffer)
{
    if (buffer.allocation == nullptr)
        return;

    VmaAllocationInfo info;
    vmaGetAllocationInfo(_allocator, buffer.allocation, &info);

    _memoryAllocations.buffers -= info.size;
}

void Device::trackTexelBuffer(const TexelBuffer &buffer)
{
    VmaAllocationInfo info;
    vmaGetAllocationInfo(_allocator, buffer.allocation, &info);

    _memoryAllocations.texelBuffers += info.size;
}

void Device::untrackTexelBuffer(const TexelBuffer &buffer)
{
    if (buffer.allocation == nullptr)
        return;

    VmaAllocationInfo info;
    vmaGetAllocationInfo(_allocator, buffer.allocation, &info);

    _memoryAllocations.texelBuffers -= info.size;
}

void Device::trackImage(const Image &image)
{
    VmaAllocationInfo info;
    vmaGetAllocationInfo(_allocator, image.allocation, &info);

    _memoryAllocations.images += info.size;
}

void Device::untrackImage(const Image &image)
{
    if (image.allocation == nullptr)
        return;

    VmaAllocationInfo info;
    vmaGetAllocationInfo(_allocator, image.allocation, &info);

    _memoryAllocations.images -= info.size;
}
