#include "Device.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>

#ifdef _WIN32
// for __debug_break()
#include <intrin.h>
#endif // _WIN32

#include <wheels/containers/hash_set.hpp>
#include <wheels/containers/static_array.hpp>
#include <wheels/containers/string.hpp>

#include "App.hpp"
#include "ForEach.hpp"
#include "ShaderReflection.hpp"
#include "Swapchain.hpp"
#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace wheels;

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
        vk::PhysicalDeviceFeatures2, features.pipelineStatisticsQuery,         \
        vk::PhysicalDeviceVulkan12Features, descriptorIndexing,                \
        vk::PhysicalDeviceVulkan12Features, descriptorBindingPartiallyBound,   \
        vk::PhysicalDeviceVulkan12Features,                                    \
        shaderSampledImageArrayNonUniformIndexing,                             \
        vk::PhysicalDeviceVulkan12Features,                                    \
        descriptorBindingUpdateUnusedWhilePending,                             \
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

bool supportsGraphics(vk::QueueFlags flags)
{
    return (flags & vk::QueueFlagBits::eGraphics) ==
           vk::QueueFlagBits::eGraphics;
}

bool supportsCompute(vk::QueueFlags flags)
{
    return (flags & vk::QueueFlagBits::eCompute) == vk::QueueFlagBits::eCompute;
}

bool supportsTransfer(vk::QueueFlags flags)
{
    return (flags & vk::QueueFlagBits::eTransfer) ==
           vk::QueueFlagBits::eTransfer;
}

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

            const vk::QueueFlags queueFlags = allFamilies[i].queueFlags;

            // Set index to matching families
            if (supportsGraphics(queueFlags))
            {
                assert(supportsCompute(queueFlags));
                assert(supportsTransfer(queueFlags));

                families.graphicsFamily = i;
                families.graphicsFamilyQueueCount = allFamilies[i].queueCount;

                if (presentSupport != VK_TRUE)
                    throw std::runtime_error(
                        "Unexpected graphics queue family without present "
                        "support. We expect to present from the graphics "
                        "queue");
            }
            else if (supportsCompute(queueFlags))
            {
                assert(supportsTransfer(queueFlags));

                families.computeFamily = i;
                families.computeFamilyQueueCount = allFamilies[i].queueCount;
            }
            else if (supportsTransfer(queueFlags))
            {
                families.transferFamily = i;
                families.transferFamilyQueueCount = allFamilies[i].queueCount;
            }
        }

        if (families.isComplete())
            break;
    }

    if (!families.computeFamily.has_value())
    {
        families.computeFamily = families.graphicsFamily;
        families.computeFamilyQueueCount = families.graphicsFamilyQueueCount;
    }
    if (!families.transferFamily.has_value())
    {
        families.transferFamily = families.computeFamily;
        families.transferFamilyQueueCount = families.computeFamilyQueueCount;
    }

    assert(
        (!families.graphicsFamily.has_value() ||
         (allFamilies[*families.graphicsFamily].timestampValidBits == 64)) &&
        "All bits assumed to be valid for simplicity in profiler");

    return families;
}

bool checkDeviceExtensionSupport(
    ScopedScratch scopeAlloc, const vk::PhysicalDevice device)
{
    const auto availableExtensions =
        device.enumerateDeviceExtensionProperties(nullptr);

    // Check that all needed extensions are present
    HashSet<String> requiredExtensions{scopeAlloc, deviceExtensions.size() * 2};
    for (const char *ext : deviceExtensions)
        requiredExtensions.insert(String{scopeAlloc, ext});

    for (const auto &extension : availableExtensions)
        requiredExtensions.remove(String{scopeAlloc, extension.extensionName});

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

Array<String> getRequiredExtensions(Allocator &alloc)
{
    // Query extensions glfw requires
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions =
        glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    Array<String> extensions{alloc, glfwExtensionCount};
    for (size_t i = 0; i < glfwExtensionCount; ++i)
        extensions.emplace_back(alloc, glfwExtensions[i]);

    // Add extension containing debug layers
    extensions.emplace_back(alloc, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    return extensions;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    const VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    const VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData)
{
    (void)messageSeverity;
    (void)messageType;

    const bool breakOnError =
        reinterpret_cast<Device::Settings *>(pUserData)->breakOnValidationError;

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

    if (breakOnError &&
        messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
#ifdef _WIN32
        // Assumes MSVC
        __debugbreak();
#else  // !__WIN32
       // Assumes gcc or a new enough clang
        __builtin_trap();
#endif // __WIN32

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

FileIncluder::FileIncluder(Allocator &alloc)
: _alloc{alloc}
, _includePath{resPath("shader")}
, _includeContent{alloc}
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
    content.path = std::make_unique<String>(
        _alloc, requestedSource.generic_string().c_str());

    content.content =
        std::make_unique<String>(readFileString(_alloc, requestedSource));

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
    _includeContent.insert_or_assign(_includeContentID++, WHEELS_MOV(content));

    return result_ptr;
}

void FileIncluder::ReleaseInclude(shaderc_include_result *data)
{
    auto id = reinterpret_cast<uint64_t>(data->user_data);
    _includeContent.remove(id);
}

Device::Device(
    Allocator &generalAlloc, ScopedScratch scopeAlloc, GLFWwindow *window,
    const Settings &settings)
: _generalAlloc{generalAlloc}
, _settings{settings}
{
    printf("Creating Vulkan device\n");

    // Use general allocator since the include set is unbounded
    _compilerOptions.SetIncluder(std::make_unique<FileIncluder>(_generalAlloc));
    _compilerOptions.SetGenerateDebugInfo();
    _compilerOptions.SetTargetSpirv(shaderc_spirv_version_1_6);
    _compilerOptions.SetTargetEnvironment(
        shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);

    const vk::DynamicLoader dl;
    auto vkGetInstanceProcAddr =
        dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    createInstance(scopeAlloc.child_scope());
    VULKAN_HPP_DEFAULT_DISPATCHER.init(_instance);

    {
        // 1.0 doesn't have the check function
        bool api_support_missing =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkEnumerateInstanceVersion == nullptr;

        if (!api_support_missing)
            api_support_missing =
                vk::enumerateInstanceVersion() < VK_VERSION_1_3;

        if (api_support_missing)
        {
            throw std::runtime_error(
                "Vulkan 1.3 required, missing support on instance");
        }
    }

    createDebugMessenger();
    createSurface(window);
    selectPhysicalDevice(scopeAlloc.child_scope());
    _queueFamilies = findQueueFamilies(_physical, _surface);

    createLogicalDevice();
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
        }

        printf("%s\n", _properties.device.deviceName.data());
    }

    assert(_transferQueue.has_value() == _transferPool.has_value());
}

Device::~Device()
{
    // Also cleans up associated command buffers
    _logical.destroy(_graphicsPool);
    if (_transferPool.has_value())
        _logical.destroy(*_transferPool);
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

Optional<vk::CommandPool> Device::transferPool() const { return _transferPool; }

Optional<vk::Queue> Device::transferQueue() const { return _transferQueue; }

const QueueFamilies &Device::queueFamilies() const { return _queueFamilies; }

const DeviceProperties &Device::properties() const { return _properties; }

wheels::Optional<Device::ShaderCompileResult> Device::compileShaderModule(
    ScopedScratch scopeAlloc, CompileShaderModuleArgs const &info)
{
    assert(info.relPath.string().starts_with("shader/"));
    const auto shaderPath = resPath(info.relPath);

    // Prepend version, defines and reset line offset before the actual source
    const String source = readFileString(scopeAlloc, shaderPath);

    const StaticArray versionLine = "#version 460\n";
    const StaticArray line1Tag = "#line 1\n";

    const size_t fullSize = versionLine.size() - 1 + line1Tag.size() - 1 +
                            info.defines.size() + source.size();
    String fullSource{scopeAlloc, fullSize};
    fullSource.extend(versionLine.data());
    fullSource.extend(info.defines);
    fullSource.extend(line1Tag.data());
    fullSource.extend(source);

    const auto result = _compiler.CompileGlslToSpv(
        fullSource.c_str(), fullSource.size(), shaderc_glsl_infer_from_source,
        shaderPath.string().c_str(), _compilerOptions);

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

    if (_settings.dumpShaderDisassembly)
    {
        const shaderc::AssemblyCompilationResult resultAsm =
            _compiler.CompileGlslToSpvAssembly(
                fullSource.c_str(), fullSource.size(),
                shaderc_glsl_infer_from_source, shaderPath.string().c_str(),
                _compilerOptions);
        if (const shaderc_compilation_status status =
                result.GetCompilationStatus();
            status == shaderc_compilation_status_success)
            fprintf(stdout, "%s\n", resultAsm.begin());
        else
        {
            const std::string err = result.GetErrorMessage();
            if (!err.empty())
                fprintf(stderr, "%s\n", err.c_str());
            fprintf(
                stderr, "Compilation of '%s' failed\n",
                shaderPath.string().c_str());
            fprintf(stderr, "%s\n", statusString(status));
            return {};
        }
    }

    const Span<const uint32_t> spvWords{
        result.begin(), asserted_cast<size_t>(result.end() - result.begin())};

    ShaderReflection reflection{
        scopeAlloc.child_scope(), _generalAlloc, spvWords};

    const auto sm = _logical.createShaderModule(vk::ShaderModuleCreateInfo{
        .codeSize = spvWords.size() * sizeof(uint32_t),
        .pCode = spvWords.data(),
    });

    _logical.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
        .objectType = vk::ObjectType::eShaderModule,
        .objectHandle =
            reinterpret_cast<uint64_t>(static_cast<VkShaderModule>(sm)),
        .pObjectName = info.debugName,
    });

    return ShaderCompileResult{
        .module = sm,
        .reflection = WHEELS_MOV(reflection),
    };
}

wheels::Optional<ShaderReflection> Device::reflectShader(
    ScopedScratch scopeAlloc, CompileShaderModuleArgs const &info,
    bool add_dummy_compute_boilerplate)
{
    assert(info.relPath.string().starts_with("shader/"));
    const auto shaderPath = resPath(info.relPath);

    // Prepend version, defines and reset line offset before the actual source
    const String source = readFileString(scopeAlloc, shaderPath);

    const StaticArray versionLine = "#version 460\n";
    const StaticArray line1Tag = "#line 1\n";

    const StaticArray computeBoilerplate1 = "#pragma shader_stage(compute)\n";
    const StaticArray computeBoilerplate2 =
        R"(
layout(local_size_x = 16, local_size_y = 16) in;
void main()
{
}
)";

    const size_t fullSize =
        versionLine.size() - 1 + line1Tag.size() - 1 + info.defines.size() +
        source.size() +
        (add_dummy_compute_boilerplate
             ? (computeBoilerplate1.size() + computeBoilerplate2.size() - 2)
             : 0);
    String fullSource{scopeAlloc, fullSize};
    fullSource.extend(versionLine.data());
    if (add_dummy_compute_boilerplate)
        fullSource.extend(computeBoilerplate1.data());
    fullSource.extend(info.defines);
    fullSource.extend(line1Tag.data());
    fullSource.extend(source);
    if (add_dummy_compute_boilerplate)
        fullSource.extend(computeBoilerplate2.data());

    const auto result = _compiler.CompileGlslToSpv(
        fullSource.c_str(), fullSource.size(), shaderc_glsl_infer_from_source,
        shaderPath.string().c_str(), _compilerOptions);

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

    const Span<const uint32_t> spvWords{
        result.begin(), asserted_cast<size_t>(result.end() - result.begin())};

    ShaderReflection reflection{
        scopeAlloc.child_scope(), _generalAlloc, spvWords};

    return WHEELS_MOV(reflection);
}

Buffer Device::create(const BufferCreateInfo &info)
{
    return createBuffer(info);
}

Buffer Device::createBuffer(const BufferCreateInfo &info)
{
    const BufferDescription &desc = info.desc;

    const vk::BufferCreateInfo bufferInfo{
        .size = desc.byteSize,
        .usage = desc.usage,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    VmaAllocationCreateFlags allocFlags = 0;
    if ((desc.properties & vk::MemoryPropertyFlagBits::eHostVisible) ==
        vk::MemoryPropertyFlagBits::eHostVisible)
        // Readback is not used yet so assume this is for staging
        allocFlags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    if (info.createMapped)
        allocFlags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;

    const VmaAllocationCreateInfo allocCreateInfo = {
        .flags = allocFlags,
        .usage = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags = static_cast<VkMemoryPropertyFlags>(desc.properties),
    };

    Buffer buffer;
    VmaAllocationInfo allocInfo;
    const auto *vkpBufferInfo =
        reinterpret_cast<const VkBufferCreateInfo *>(&bufferInfo);
    auto *vkpBuffer = reinterpret_cast<VkBuffer *>(&buffer.handle);
    // Just align to the maximum requirement that's out in the wild (AMD with
    // some drivers). Small buffers should be few anyway so if the memory lost
    // to alignment ends up being a problem, the fix is likely to not have so
    // many individual buffers
    const vk::DeviceSize alignment = 256;
    {
        const std::lock_guard _lock{_allocatorMutex};
        vmaCreateBufferWithAlignment(
            _allocator, vkpBufferInfo, &allocCreateInfo, alignment, vkpBuffer,
            &buffer.allocation, &allocInfo);
    }

    buffer.byteSize = desc.byteSize;

    if (info.createMapped)
    {
        assert(allocInfo.pMappedData);
        buffer.mapped = allocInfo.pMappedData;
    }

    _logical.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
        .objectType = vk::ObjectType::eBuffer,
        .objectHandle =
            reinterpret_cast<uint64_t>(static_cast<VkBuffer>(buffer.handle)),
        .pObjectName = info.debugName,
    });

    if (info.initialData != nullptr)
    {
        const StaticArray postfix = "StagingBuffer";
        String stagingDebugName{
            _generalAlloc, strlen(info.debugName) + postfix.size() - 1};
        stagingDebugName.extend(info.debugName);
        stagingDebugName.extend(postfix.data());

        const auto stagingBuffer = createBuffer(BufferCreateInfo{
            .desc =
                BufferDescription{
                    .byteSize = desc.byteSize,
                    .usage = vk::BufferUsageFlagBits::eTransferSrc,
                    .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                                  vk::MemoryPropertyFlagBits::eHostCoherent,
                },
            .createMapped = true,
            .debugName = stagingDebugName.c_str(),
        });

        memcpy(stagingBuffer.mapped, info.initialData, desc.byteSize);

        const auto commandBuffer = beginGraphicsCommands();

        const vk::BufferCopy copyRegion{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = desc.byteSize,
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
    {
        const std::lock_guard _lock{_allocatorMutex};
        vmaDestroyBuffer(_allocator, vkBuffer, buffer.allocation);
    }
}

TexelBuffer Device::create(const TexelBufferCreateInfo &info)
{
    return createTexelBuffer(info);
}

TexelBuffer Device::createTexelBuffer(const TexelBufferCreateInfo &info)
{
    const TexelBufferDescription &desc = info.desc;
    const BufferDescription &bufferDesc = desc.bufferDesc;

    const auto formatProperties = _physical.getFormatProperties(desc.format);

    if (containsFlag(
            bufferDesc.usage, vk::BufferUsageFlagBits::eStorageTexelBuffer))
    {
        assertContainsFlag(
            formatProperties.bufferFeatures,
            vk::FormatFeatureFlagBits::eStorageTexelBuffer,
            "Format doesn't support storage texel buffer");
    }
    if (containsFlag(
            bufferDesc.usage, vk::BufferUsageFlagBits::eUniformTexelBuffer))
    {
        assertContainsFlag(
            formatProperties.bufferFeatures,
            vk::FormatFeatureFlagBits::eUniformTexelBuffer,
            "Format doesn't support uniform texel buffer");
    }
    if (desc.supportAtomics)
    {
        assertContainsFlag(
            formatProperties.bufferFeatures,
            vk::FormatFeatureFlagBits::eStorageTexelBufferAtomic,
            "Format doesn't support atomics");
    }

    const auto buffer = createBuffer(BufferCreateInfo{
        .desc = bufferDesc,
        .debugName = info.debugName,
    });
    // This will be tracked as a texel buffer
    untrackBuffer(buffer);

    const auto view = _logical.createBufferView(vk::BufferViewCreateInfo{
        .buffer = buffer.handle,
        .format = desc.format,
        .offset = 0,
        .range = bufferDesc.byteSize,
    });

    const TexelBuffer ret{
        .handle = buffer.handle,
        .view = view,
        .format = desc.format,
        .size = bufferDesc.byteSize,
        .allocation = buffer.allocation,
    };
    trackTexelBuffer(ret);

    return ret;
}

void Device::destroy(const TexelBuffer &buffer)
{
    untrackTexelBuffer(buffer);

    auto *vkBuffer = static_cast<VkBuffer>(buffer.handle);
    {
        const std::lock_guard _lock{_allocatorMutex};
        vmaDestroyBuffer(_allocator, vkBuffer, buffer.allocation);
    }
    _logical.destroy(buffer.view);
}

Image Device::create(const ImageCreateInfo &info) { return createImage(info); }

Image Device::createImage(const ImageCreateInfo &info)
{
    const ImageDescription &desc = info.desc;

    const vk::Extent3D extent{
        .width = desc.width,
        .height = desc.height,
        .depth = desc.depth,
    };

    const vk::ImageCreateInfo imageInfo{
        .flags = desc.createFlags,
        .imageType = desc.imageType,
        .format = desc.format,
        .extent = extent,
        .mipLevels = desc.mipCount,
        .arrayLayers = desc.layerCount,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = desc.usageFlags,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const VmaAllocationCreateInfo allocInfo = {
        .flags = {}, // Device only
        .usage = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags = static_cast<VkMemoryPropertyFlags>(desc.properties),
    };

    Image image;
    const auto *vkpImageInfo =
        reinterpret_cast<const VkImageCreateInfo *>(&imageInfo);
    auto *vkpImage = reinterpret_cast<VkImage *>(&image.handle);
    {
        const std::lock_guard _lock{_allocatorMutex};
        vmaCreateImage(
            _allocator, vkpImageInfo, &allocInfo, vkpImage, &image.allocation,
            nullptr);
    }

    _logical.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
        .objectType = vk::ObjectType::eImage,
        .objectHandle =
            reinterpret_cast<uint64_t>(static_cast<VkImage>(image.handle)),
        .pObjectName = info.debugName,
    });

    const vk::ImageSubresourceRange range{
        .aspectMask = aspectMask(desc.format),
        .baseMipLevel = 0,
        .levelCount = desc.mipCount,
        .baseArrayLayer = 0,
        .layerCount = desc.layerCount,
    };

    const vk::ImageViewType viewType = [desc]()
    {
        switch (desc.imageType)
        {
        case vk::ImageType::e1D:
            if (desc.layerCount == 1)
                return vk::ImageViewType::e1D;
            else
                return vk::ImageViewType::e1DArray;
        case vk::ImageType::e2D:
            if (desc.layerCount == 1)
                return vk::ImageViewType::e2D;
            else
            {
                if ((desc.createFlags &
                     vk::ImageCreateFlagBits::eCubeCompatible) ==
                    vk::ImageCreateFlagBits::eCubeCompatible)
                {
                    assert(desc.layerCount == 6 && "Cube arrays not supported");
                    return vk::ImageViewType::eCube;
                }
                return vk::ImageViewType::e2DArray;
            }
        case vk::ImageType::e3D:
            assert(desc.layerCount == 1 && "Can't have 3D image arrays");
            return vk::ImageViewType::e3D;
        default:
            throw std::runtime_error(
                "Unexpected image type " + to_string(desc.imageType));
        }
    }();

    image.view = _logical.createImageView(vk::ImageViewCreateInfo{
        .image = image.handle,
        .viewType = viewType,
        .format = desc.format,
        .subresourceRange = range,
    });

    image.extent = extent;
    image.subresourceRange = range;
    image.imageType = desc.imageType;
    image.format = desc.format;

    {
        VmaAllocationInfo vmaInfo;
        {
            const std::lock_guard _lock{_allocatorMutex};
            vmaGetAllocationInfo(_allocator, image.allocation, &vmaInfo);
        }
        image.rawByteSize = vmaInfo.size;
    }

    trackImage(image);

    return image;
}

void Device::destroy(const Image &image)
{
    untrackImage(image);

    auto *vkImage = static_cast<VkImage>(image.handle);
    {
        const std::lock_guard _lock{_allocatorMutex};
        vmaDestroyImage(_allocator, vkImage, image.allocation);
    }
    _logical.destroy(image.view);
}

void Device::createSubresourcesViews(
    const Image &image, Span<vk::ImageView> outViews) const
{
    assert(
        image.subresourceRange.layerCount == 1 &&
        "Texture arrays not supported");
    assert(
        image.subresourceRange.levelCount > 1 &&
        "You can just use the global view when no mips are present");
    assert(image.subresourceRange.baseMipLevel == 0);
    assert(image.subresourceRange.levelCount == outViews.size());

    const vk::ImageAspectFlags aspect = aspectMask(image.format);
    const vk::ImageViewType viewType = [&image]()
    {
        switch (image.imageType)
        {
        case vk::ImageType::e1D:
            return vk::ImageViewType::e1D;
        case vk::ImageType::e2D:
            return vk::ImageViewType::e2D;
        case vk::ImageType::e3D:
            return vk::ImageViewType::e3D;
        default:
            throw std::runtime_error(
                "Unexpected image type " + to_string(image.imageType));
        }
    }();

    for (uint32_t i = 0; i < image.subresourceRange.levelCount; ++i)
        outViews[i] = _logical.createImageView(vk::ImageViewCreateInfo{
            .image = image.handle,
            .viewType = viewType,
            .format = image.format,
            .subresourceRange =
                vk::ImageSubresourceRange{
                    .aspectMask = aspect,
                    .baseMipLevel = i,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        });
}

void Device::destroy(Span<const vk::ImageView> views) const
{
    for (const vk::ImageView view : views)
        _logical.destroy(view);
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

bool Device::isDeviceSuitable(
    ScopedScratch scopeAlloc, const vk::PhysicalDevice device) const
{
    const auto families = findQueueFamilies(device, _surface);
    if (!families.isComplete())
    {
        fprintf(stderr, "Missing required queue families\n");
        return false;
    }

    if (!checkDeviceExtensionSupport(scopeAlloc.child_scope(), device))
        return false;

    const SwapchainSupport swapSupport{scopeAlloc, device, _surface};
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

    const auto props = device.getProperties2<
        vk::PhysicalDeviceProperties2, vk::PhysicalDeviceSubgroupProperties>();

    {
        const vk::PhysicalDeviceProperties2 &deviceProps =
            props.get<vk::PhysicalDeviceProperties2>();

        if (deviceProps.properties.apiVersion < VK_VERSION_1_3)
        {
            fprintf(stderr, "Missing Vulkan 1.3 support\n");
            return false;
        }
    }

    {
        const vk::PhysicalDeviceSubgroupProperties &subgroupProps =
            props.get<vk::PhysicalDeviceSubgroupProperties>();

        // Vulkan 1.1 requires
        //   - support in compute
        //   - basic ops

        if ((subgroupProps.supportedOperations &
             vk::SubgroupFeatureFlagBits::eArithmetic) !=
            vk::SubgroupFeatureFlagBits::eArithmetic)
        {
            fprintf(stderr, "Missing subgroup arithmetic op support\n");
            return false;
        }
    }

    printf("Required features are supported\n");

    return true;
}

void Device::createInstance(ScopedScratch scopeAlloc)
{
    if (_settings.enableDebugLayers && !checkValidationLayerSupport())
        throw std::runtime_error("Validation layers not available");

    const vk::ApplicationInfo appInfo{
        .pApplicationName = "prosper",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "prosper",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_3,
    };

    const Array<String> extensions = getRequiredExtensions(scopeAlloc);

    Array<const char *> extension_cstrs{scopeAlloc, extensions.size()};
    for (const String &ext : extensions)
        extension_cstrs.push_back(ext.c_str());

    _instance = vk::createInstance(vk::InstanceCreateInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount =
            _settings.enableDebugLayers
                ? asserted_cast<uint32_t>(validationLayers.size())
                : 0,
        .ppEnabledLayerNames =
            _settings.enableDebugLayers ? validationLayers.data() : nullptr,
        .enabledExtensionCount =
            asserted_cast<uint32_t>(extension_cstrs.size()),
        .ppEnabledExtensionNames = extension_cstrs.data(),
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
        .pUserData = &_settings,
    };
    CreateDebugUtilsMessengerEXT(
        _instance, &createInfo, nullptr, &_debugMessenger);
}

void Device::createSurface(GLFWwindow *window)
{
    assert(window != nullptr);

    auto *vkpSurface = reinterpret_cast<VkSurfaceKHR *>(&_surface);
    auto *vkInstance = static_cast<VkInstance>(_instance);
    if (glfwCreateWindowSurface(vkInstance, window, nullptr, vkpSurface) !=
        VK_SUCCESS)
        throw std::runtime_error("Failed to create window surface");
}

void Device::selectPhysicalDevice(ScopedScratch scopeAlloc)
{
    printf("Selecting device\n");

    const auto devices = _instance.enumeratePhysicalDevices();

    for (const auto &device : devices)
    {
        printf("Considering '%s'\n", device.getProperties().deviceName.data());
        if (isDeviceSuitable(scopeAlloc.child_scope(), device))
        {
            _physical = device;
            return;
        }
    }

    throw std::runtime_error("Failed to find a suitable GPU");
}

void Device::createLogicalDevice()
{
    assert(_queueFamilies.graphicsFamily.has_value());
    assert(_queueFamilies.transferFamily.has_value());

    const uint32_t graphicsFamily = *_queueFamilies.graphicsFamily;
    const uint32_t graphicsFamilyQueueCount =
        _queueFamilies.graphicsFamilyQueueCount;
    const uint32_t transferFamily = *_queueFamilies.transferFamily;

    // First queue in family has largest queue, rest descend
    const StaticArray queuePriorities = {1.f, 0.f};
    const StaticArray<vk::DeviceQueueCreateInfo, 2> queueCreateInfos = [&]
    {
        StaticArray<vk::DeviceQueueCreateInfo, 2> cis;
        if (graphicsFamily == transferFamily)
        {
            assert(queuePriorities.size() >= 2);
            cis.push_back(vk::DeviceQueueCreateInfo{
                .queueFamilyIndex = graphicsFamily,
                .queueCount = std::min(2u, graphicsFamilyQueueCount),
                .pQueuePriorities = queuePriorities.data(),
            });
        }
        else
        {
            cis.push_back(vk::DeviceQueueCreateInfo{
                .queueFamilyIndex = graphicsFamily,
                .queueCount = 1,
                .pQueuePriorities = queuePriorities.data(),
            });
            cis.push_back(vk::DeviceQueueCreateInfo{
                .queueFamilyIndex = transferFamily,
                .queueCount = 1,
                .pQueuePriorities = queuePriorities.data(),
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
            _settings.enableDebugLayers
                ? asserted_cast<uint32_t>(validationLayers.size())
                : 0,
        .ppEnabledLayerNames =
            _settings.enableDebugLayers ? validationLayers.data() : nullptr,
        .enabledExtensionCount =
            asserted_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
    };

#define TOGGLE_REQUIRED_FEATURES(container, feature)                           \
    createChain.get<container>().feature = VK_TRUE;

    FOR_EACH_PAIR(TOGGLE_REQUIRED_FEATURES, REQUIRED_FEATURES);

#undef TOGGLE_REQUIRED_FEATURES

    _logical = _physical.createDevice(createChain.get<vk::DeviceCreateInfo>());

    _graphicsQueue = _logical.getQueue(graphicsFamily, 0);
    if (graphicsFamily == transferFamily)
    {
        if (graphicsFamilyQueueCount > 1)
            _transferQueue = _logical.getQueue(graphicsFamily, 1);
        // No separate transfer queue if it couldn't be created from the
        // graphics family
    }
    else
        _transferQueue = _logical.getQueue(transferFamily, 0);
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
        assert(_queueFamilies.graphicsFamily.has_value());

        const vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = *_queueFamilies.graphicsFamily,
        };
        _graphicsPool = _logical.createCommandPool(poolInfo, nullptr);
    }
    {
        if (_transferQueue.has_value())
        {
            assert(_queueFamilies.transferFamily.has_value());

            const vk::CommandPoolCreateInfo poolInfo{
                .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                .queueFamilyIndex = *_queueFamilies.transferFamily,
            };
            _transferPool = _logical.createCommandPool(poolInfo, nullptr);
        }
    }
}

void Device::trackBuffer(const Buffer &buffer)
{
    VmaAllocationInfo info;
    {
        const std::lock_guard _lock{_allocatorMutex};
        vmaGetAllocationInfo(_allocator, buffer.allocation, &info);
    }

    _memoryAllocations.buffers += info.size;
}

void Device::untrackBuffer(const Buffer &buffer)
{
    if (buffer.allocation == nullptr)
        return;

    VmaAllocationInfo info;
    {
        const std::lock_guard _lock{_allocatorMutex};
        vmaGetAllocationInfo(_allocator, buffer.allocation, &info);
    }

    _memoryAllocations.buffers -= info.size;
}

void Device::trackTexelBuffer(const TexelBuffer &buffer)
{
    VmaAllocationInfo info;
    {
        const std::lock_guard _lock{_allocatorMutex};
        vmaGetAllocationInfo(_allocator, buffer.allocation, &info);
    }

    _memoryAllocations.texelBuffers += info.size;
}

void Device::untrackTexelBuffer(const TexelBuffer &buffer)
{
    if (buffer.allocation == nullptr)
        return;

    VmaAllocationInfo info;
    {
        const std::lock_guard _lock{_allocatorMutex};
        vmaGetAllocationInfo(_allocator, buffer.allocation, &info);
    }

    _memoryAllocations.texelBuffers -= info.size;
}

void Device::trackImage(const Image &image)
{
    VmaAllocationInfo info;
    {
        const std::lock_guard _lock{_allocatorMutex};
        vmaGetAllocationInfo(_allocator, image.allocation, &info);
    }

    _memoryAllocations.images += info.size;
}

void Device::untrackImage(const Image &image)
{
    if (image.allocation == nullptr)
        return;

    VmaAllocationInfo info;
    {
        const std::lock_guard _lock{_allocatorMutex};
        vmaGetAllocationInfo(_allocator, image.allocation, &info);
    }

    _memoryAllocations.images -= info.size;
}
