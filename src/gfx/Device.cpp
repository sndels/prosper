#include "Device.hpp"

#include <cinttypes>
#include <cstring>
#include <iostream>
#include <stdexcept>

#ifdef _WIN32
// for __debug_break()
#include <intrin.h>
#endif // _WIN32

#include "Allocators.hpp"
#include "Window.hpp"
#include "gfx/ShaderIncludes.hpp"
#include "gfx/ShaderReflection.hpp"
#include "gfx/Swapchain.hpp"
#include "gfx/VkUtils.hpp"
#include "utils/ForEach.hpp"
#include "utils/Logger.hpp"
#include "utils/Utils.hpp"

#include <GLFW/glfw3.h>
#include <wheels/containers/hash_map.hpp>
#include <wheels/containers/inline_array.hpp>
#include <wheels/containers/static_array.hpp>
#include <wheels/containers/string.hpp>
#include <wheels/owning_ptr.hpp>

using namespace wheels;

#define ALL_FEATURE_STRUCTS_LIST                                               \
    vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,           \
        vk::PhysicalDeviceVulkan12Features,                                    \
        vk::PhysicalDeviceVulkan13Features,                                    \
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR,                    \
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,                       \
        vk::PhysicalDeviceMeshShaderFeaturesEXT

#define REQUIRED_FEATURES                                                      \
    vk::PhysicalDeviceFeatures2, features.geometryShader,                      \
        vk::PhysicalDeviceFeatures2, features.samplerAnisotropy,               \
        vk::PhysicalDeviceFeatures2,                                           \
        features.shaderStorageImageReadWithoutFormat,                          \
        vk::PhysicalDeviceFeatures2,                                           \
        features.shaderStorageImageWriteWithoutFormat,                         \
        vk::PhysicalDeviceFeatures2,                                           \
        features.shaderSampledImageArrayDynamicIndexing,                       \
        vk::PhysicalDeviceFeatures2, features.pipelineStatisticsQuery,         \
        vk::PhysicalDeviceVulkan11Features, storageBuffer16BitAccess,          \
        vk::PhysicalDeviceVulkan12Features, descriptorIndexing,                \
        vk::PhysicalDeviceVulkan12Features, descriptorBindingPartiallyBound,   \
        vk::PhysicalDeviceVulkan12Features,                                    \
        shaderSampledImageArrayNonUniformIndexing,                             \
        vk::PhysicalDeviceVulkan12Features,                                    \
        shaderStorageBufferArrayNonUniformIndexing,                            \
        vk::PhysicalDeviceVulkan12Features,                                    \
        descriptorBindingUpdateUnusedWhilePending,                             \
        vk::PhysicalDeviceVulkan12Features,                                    \
        descriptorBindingVariableDescriptorCount,                              \
        vk::PhysicalDeviceVulkan12Features, runtimeDescriptorArray,            \
        vk::PhysicalDeviceVulkan12Features, hostQueryReset,                    \
        vk::PhysicalDeviceVulkan12Features, bufferDeviceAddress,               \
        vk::PhysicalDeviceVulkan12Features, storageBuffer8BitAccess,           \
        vk::PhysicalDeviceVulkan13Features, synchronization2,                  \
        vk::PhysicalDeviceVulkan13Features, dynamicRendering,                  \
        vk::PhysicalDeviceVulkan13Features, maintenance4,                      \
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR,                    \
        accelerationStructure,                                                 \
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR, rayTracingPipeline,   \
        vk::PhysicalDeviceMeshShaderFeaturesEXT, meshShader

namespace
{

const uint64_t sShaderCacheMagic = 0x4448'5352'5053'5250; // PRSPRSHD
// This should be incremented when breaking changes are made to what's cached or
// when the shader compiler is updated
const uint32_t sShaderCacheVersion = 2;

const char *const sCppStyleLineDirectiveCStr =
    "#extension GL_GOOGLE_cpp_style_line_directive : require\n";
const StrSpan sCppStyleLineDirective{sCppStyleLineDirectiveCStr};

constexpr std::array validationLayers = {
    //"VK_LAYER_LUNARG_api_dump",
    "VK_LAYER_KHRONOS_validation",
};
constexpr StaticArray deviceExtensions{{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_EXT_MESH_SHADER_EXTENSION_NAME,
}};

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
                WHEELS_ASSERT(supportsCompute(queueFlags));
                WHEELS_ASSERT(supportsTransfer(queueFlags));

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
                WHEELS_ASSERT(supportsTransfer(queueFlags));

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

    WHEELS_ASSERT(
        (!families.graphicsFamily.has_value() ||
         (allFamilies[*families.graphicsFamily].timestampValidBits == 64)) &&
        "All bits assumed to be valid for simplicity in profiler");

    return families;
}

bool checkDeviceExtensionSupport(
    ScopedScratch scopeAlloc, const vk::PhysicalDevice device,
    const Device::Settings &settings)
{
    const auto availableExtensions =
        device.enumerateDeviceExtensionProperties(nullptr);

    // Check that all needed extensions are present
    HashSet<String> requiredExtensions{scopeAlloc, deviceExtensions.size() * 2};
    for (const char *ext : deviceExtensions)
        requiredExtensions.insert(String{scopeAlloc, ext});

    if (settings.robustAccess)
        requiredExtensions.insert(
            String{scopeAlloc, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME});

    for (const auto &extension : availableExtensions)
        requiredExtensions.remove(String{scopeAlloc, extension.extensionName});

    if (!requiredExtensions.empty())
    {
        LOG_ERR("Missing support for extensions:");
        for (const auto &e : requiredExtensions)
            LOG_ERR("  %s", e.c_str());
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
    (void)messageType;

    const bool breakOnError =
        static_cast<Device::Settings *>(pUserData)->breakOnValidationError;

    // VK_TRUE is reserved
    constexpr auto ret = VK_FALSE;

#define DEVICE_EXTENSION_STR "Device Extension: "
    // Skip extension dump noise
    if (strncmp(
            DEVICE_EXTENSION_STR, pCallbackData->pMessage,
            sizeof(DEVICE_EXTENSION_STR) - 1) == 0)
        return ret;
#undef DEVICE_EXTENSION_STR

    {
        // Suppress a false positive in validation layers 1.3.283
        // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/8038
        // This can be removed when the next version drops
        const String tmp{gAllocators.general, pCallbackData->pMessage};
        if (tmp.contains(
                "[RtDiTrace] doesn't set up VK_DYNAMIC_STATE_VIEWPORT"))
            return ret;
    }

    if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0)
        LOG_ERR("VkDbg: %s", pCallbackData->pMessage);
    else if (
        (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) !=
        0)
        LOG_WARN("VkDbg: %s", pCallbackData->pMessage);
    else
        LOG_INFO("VkDbg: %s", pCallbackData->pMessage);

    if (breakOnError &&
        messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        WHEELS_DEBUGBREAK();

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

// Returns true if the cache is valid
// No ScopedScratch because spvWords is already scoped in the upper scope and a
// child scope would stomp it.
bool readCache(
    Allocator &alloc, const std::filesystem::path &cachePath,
    Array<uint32_t> *spvWords = nullptr,
    HashSet<std::filesystem::path> *uniqueIncludes = nullptr)
{
    if (!std::filesystem::exists(cachePath))
        return false;

    std::ifstream cacheFile{cachePath, std::ios_base::binary};

    uint64_t magic{0};
    static_assert(sizeof(magic) == sizeof(sShaderCacheMagic));

    readRaw(cacheFile, magic);
    if (magic != sShaderCacheMagic)
        throw std::runtime_error(
            "Expected a valid shader cache in file '" + cachePath.string() +
            "'");

    uint32_t version{0};
    static_assert(sizeof(version) == sizeof(sShaderCacheVersion));
    readRaw(cacheFile, version);
    if (version != sShaderCacheVersion)
        return false;

    WHEELS_ASSERT(
        (spvWords == nullptr && uniqueIncludes == nullptr) ||
        (spvWords != nullptr && uniqueIncludes != nullptr));
    if (spvWords == nullptr)
        return true;

    uint32_t includeCount{0};
    readRaw(cacheFile, includeCount);
    for (uint32_t i = 0; i < includeCount; ++i)
    {
        uint32_t includeLength{0};
        readRaw(cacheFile, includeLength);

        // Reserve room for null terminated but read without null
        Array<char> include{alloc, includeLength + 1};
        include.resize(includeLength);
        readRawSpan(cacheFile, include.mut_span());
        include.push_back('\0');

        uniqueIncludes->insert(std::filesystem::path{include.data()});
    }

    uint32_t spvWordCount{0};
    readRaw(cacheFile, spvWordCount);

    spvWords->resize(spvWordCount);
    readRawSpan(cacheFile, spvWords->mut_span());

    return true;
}

void writeCache(
    const std::filesystem::path &cachePath,
    const shaderc::SpvCompilationResult &compilationResult,
    const HashSet<std::filesystem::path> &uniqueIncludes)
{
    const std::filesystem::path parentFolder = cachePath.parent_path();
    if (!std::filesystem::exists(parentFolder))
        std::filesystem::create_directories(parentFolder);

    std::filesystem::remove(cachePath);

    // Write into a tmp file and rename when done to minimize the potential for
    // corrupted files
    std::filesystem::path cacheTmpPath = cachePath;
    cacheTmpPath.replace_extension("prosper_shader_TMP");

    std::ofstream cacheFile{cacheTmpPath, std::ios_base::binary};

    writeRaw(cacheFile, sShaderCacheMagic);
    writeRaw(cacheFile, sShaderCacheVersion);
    writeRaw(cacheFile, asserted_cast<uint32_t>(uniqueIncludes.size()));
    for (const std::filesystem::path &include : uniqueIncludes)
    {
        // This has to match what recompiles compare against because of how path
        // hashing works
        const std::string genericPath = include.lexically_normal().string();
        writeRaw(cacheFile, asserted_cast<uint32_t>(genericPath.size()));
        writeRawStrSpan(
            cacheFile, StrSpan{genericPath.c_str(), genericPath.size()});
    }
    const size_t spvWordCount =
        compilationResult.end() - compilationResult.begin();
    writeRaw(cacheFile, asserted_cast<uint32_t>(spvWordCount));
    writeRawSpan(cacheFile, Span{compilationResult.begin(), spvWordCount});

    cacheFile.close();

    // Make sure we have rw permissions for the user to be nice
    const std::filesystem::perms initialPerms =
        std::filesystem::status(cacheTmpPath).permissions();
    std::filesystem::permissions(
        cacheTmpPath, initialPerms | std::filesystem::perms::owner_read |
                          std::filesystem::perms::owner_write);

    // Rename when the file is done to minimize the potential of a corrupted
    // file
    std::filesystem::rename(cacheTmpPath, cachePath);
}

} // namespace

// This used everywhere and init()/destroy() order relative to other similar
// globals is handled in main()
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
Device gDevice;

Device::~Device()
{
    WHEELS_ASSERT((!m_initialized || !m_instance) && "destroy() not called");
}

void Device::init(wheels::ScopedScratch scopeAlloc, Settings const &settings)
{
    WHEELS_ASSERT(!m_initialized);

    LOG_INFO("Creating Vulkan device");

    m_settings = settings;

    // No includer as we expand those ourselves
    m_compilerOptions.SetGenerateDebugInfo();
    m_compilerOptions.SetTargetSpirv(shaderc_spirv_version_1_6);
    m_compilerOptions.SetTargetEnvironment(
        shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);

    m_compiler = OwningPtr<shaderc::Compiler>(gAllocators.general);

    const vk::DynamicLoader dl;
    auto vkGetInstanceProcAddr =
        dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    createInstance(scopeAlloc.child_scope());
    VULKAN_HPP_DEFAULT_DISPATCHER.init(m_instance);

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
    createSurface();
    selectPhysicalDevice(scopeAlloc.child_scope());
    m_queueFamilies = findQueueFamilies(m_physical, m_surface);

    createLogicalDevice(scopeAlloc.child_scope());
    VULKAN_HPP_DEFAULT_DISPATCHER.init(m_logical);

    createAllocator();
    createCommandPools();

    {
        const auto props = m_physical.getProperties2<
            vk::PhysicalDeviceProperties2,
            vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
            vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
            vk::PhysicalDeviceMeshShaderPropertiesEXT,
            vk::PhysicalDeviceSubgroupProperties>();
        m_properties.device =
            props.get<vk::PhysicalDeviceProperties2>().properties;
        m_properties.rtPipeline =
            props.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
        m_properties.accelerationStructure =
            props.get<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();
        m_properties.meshShader =
            props.get<vk::PhysicalDeviceMeshShaderPropertiesEXT>();
        m_properties.subgroup =
            props.get<vk::PhysicalDeviceSubgroupProperties>();

#ifdef __linux__
        // The AMD 680M on amdpro drivers seems to misreport this higher
        // than what's actually used
        if (m_properties.device.vendorID == 0x1002 &&
            m_properties.device.deviceID == 0x1681)
            m_properties.meshShader.maxMeshWorkGroupCount[0] = std::min(
                m_properties.meshShader.maxMeshWorkGroupCount[0], 0xFFFFu);
#endif // __linux__

        WHEELS_ASSERT(
            m_properties.meshShader.maxMeshOutputVertices >= sMaxMsVertices);
        WHEELS_ASSERT(
            m_properties.meshShader.maxMeshOutputPrimitives >= sMaxMsTriangles);

        const vk::ShaderStageFlags meshAndComputeStages =
            vk::ShaderStageFlagBits::eMeshEXT |
            vk::ShaderStageFlagBits::eCompute;
        WHEELS_ASSERT(
            (m_properties.subgroup.supportedStages & meshAndComputeStages) ==
            meshAndComputeStages);

        const vk::SubgroupFeatureFlags basicBallotAndArithmetic =
            vk::SubgroupFeatureFlagBits::eBasic |
            vk::SubgroupFeatureFlagBits::eBallot |
            vk::SubgroupFeatureFlagBits::eArithmetic;
        WHEELS_ASSERT(
            (m_properties.subgroup.supportedOperations &
             basicBallotAndArithmetic) == basicBallotAndArithmetic);

        {
            const auto apiPacked = m_properties.device.apiVersion;
            const auto major = VK_API_VERSION_MAJOR(apiPacked);
            const auto minor = VK_API_VERSION_MINOR(apiPacked);
            const auto patch = VK_API_VERSION_PATCH(apiPacked);
            LOG_INFO("Vulkan %u.%u.%u", major, minor, patch);
        }

        LOG_INFO("%s", m_properties.device.deviceName.data());
    }

    m_initialized = true;
}

void Device::destroy()
{
    // Don't check for initialized as we might be cleaning up after a partial
    // init that failed

    if (m_allocator != nullptr)
        vmaDestroyAllocator(m_allocator);

    if (m_logical)
    {
        // Also cleans up associated command buffers
        m_logical.destroy(m_graphicsPool);
        m_logical.destroy(m_transferPool);
        // Implicitly cleans up associated queues as well
        m_logical.destroy();
    }

    if (m_instance)
    {
        m_instance.destroy(m_surface);
        DestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
        m_instance.destroy();

        // m_initialized = true and null instance mark a destroyed Device
        m_instance = vk::Instance{};
    }

    m_compiler.reset();
}

vk::Instance Device::instance() const
{
    WHEELS_ASSERT(m_instance);

    return m_instance;
}

vk::PhysicalDevice Device::physical() const
{
    WHEELS_ASSERT(m_physical);

    return m_physical;
}

vk::Device Device::logical() const
{
    WHEELS_ASSERT(m_logical);

    return m_logical;
}

vk::SurfaceKHR Device::surface() const
{
    WHEELS_ASSERT(m_surface);

    return m_surface;
}

vk::CommandPool Device::graphicsPool() const
{
    WHEELS_ASSERT(m_graphicsPool);

    return m_graphicsPool;
}

vk::Queue Device::graphicsQueue() const
{
    WHEELS_ASSERT(m_graphicsQueue);

    return m_graphicsQueue;
}

vk::CommandPool Device::transferPool() const
{
    WHEELS_ASSERT(m_transferPool);

    return m_transferPool;
}

vk::Queue Device::transferQueue() const
{
    WHEELS_ASSERT(m_transferQueue);

    return m_transferQueue;
}

const QueueFamilies &Device::queueFamilies() const { return m_queueFamilies; }

const DeviceProperties &Device::properties() const { return m_properties; }

wheels::Optional<Device::ShaderCompileResult> Device::compileShaderModule(
    ScopedScratch scopeAlloc, CompileShaderModuleArgs const &info)
{
    WHEELS_ASSERT(m_initialized);

    WHEELS_ASSERT(info.relPath.string().starts_with("shader/"));
    const auto shaderPath = resPath(info.relPath);

    // Prepend version, defines and reset line offset before the actual source
    const String source = readFileString(scopeAlloc, shaderPath);

    const StaticArray versionLine = "#version 460\n";
    const StaticArray line1Tag = "#line 1\n";

    const size_t fullSize = versionLine.size() - 1 + line1Tag.size() - 1 +
                            info.defines.size() + source.size();
    String topLevelSource{scopeAlloc, fullSize};
    topLevelSource.extend(versionLine.data());
    // The custom includer uses these to make errors work
    topLevelSource.extend(sCppStyleLineDirective);
    topLevelSource.extend(info.defines);
    topLevelSource.extend(line1Tag.data());
    topLevelSource.extend(source);

    const std::filesystem::path cachePath =
        updateShaderCache(scopeAlloc, shaderPath, topLevelSource, info.relPath);
    if (cachePath.empty())
        return {};

    // Always read from the cache to make caching issues always visible
    HashSet<std::filesystem::path> uniqueIncludes{scopeAlloc};
    Array<uint32_t> spvWords{scopeAlloc};
    readCache(scopeAlloc, cachePath, &spvWords, &uniqueIncludes);
    WHEELS_ASSERT(!spvWords.empty());

    ShaderReflection reflection;
    reflection.init(scopeAlloc.child_scope(), spvWords, uniqueIncludes);

    const auto sm = m_logical.createShaderModule(vk::ShaderModuleCreateInfo{
        .codeSize = spvWords.size() * sizeof(uint32_t),
        .pCode = spvWords.data(),
    });

    m_logical.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
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
    WHEELS_ASSERT(m_initialized);

    LOG_INFO("Reflecting %s", info.relPath.string().c_str());

    WHEELS_ASSERT(info.relPath.string().starts_with("shader/"));
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
    String topLevelSource{scopeAlloc, fullSize};
    topLevelSource.extend(versionLine.data());
    // The custom includer uses these to make errors work
    topLevelSource.extend(sCppStyleLineDirective);
    if (add_dummy_compute_boilerplate)
        topLevelSource.extend(computeBoilerplate1.data());
    topLevelSource.extend(info.defines);
    topLevelSource.extend(line1Tag.data());
    topLevelSource.extend(source);
    if (add_dummy_compute_boilerplate)
        topLevelSource.extend(computeBoilerplate2.data());

    const std::filesystem::path cachePath =
        updateShaderCache(scopeAlloc, shaderPath, topLevelSource, info.relPath);

    // Always read from the cache to make caching issues always visible
    HashSet<std::filesystem::path> uniqueIncludes{scopeAlloc};
    Array<uint32_t> spvWords{scopeAlloc};
    readCache(scopeAlloc, cachePath, &spvWords, &uniqueIncludes);
    WHEELS_ASSERT(!spvWords.empty());

    ShaderReflection reflection;
    reflection.init(scopeAlloc.child_scope(), spvWords, uniqueIncludes);

    return WHEELS_MOV(reflection);
}

Buffer Device::create(const BufferCreateInfo &info)
{
    WHEELS_ASSERT(m_initialized);

    return createBuffer(info);
}

Buffer Device::createBuffer(const BufferCreateInfo &info)
{
    WHEELS_ASSERT(m_initialized);

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

    const vk::MemoryPropertyFlags hostVisibleCoherent =
        (vk::MemoryPropertyFlagBits::eHostVisible |
         vk::MemoryPropertyFlagBits::eHostCoherent);
    const bool createMapped =
        (desc.properties & hostVisibleCoherent) == hostVisibleCoherent;
    if (createMapped)
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
        const std::lock_guard lock{m_allocatorMutex};
        vmaCreateBufferWithAlignment(
            m_allocator, vkpBufferInfo, &allocCreateInfo, alignment, vkpBuffer,
            &buffer.allocation, &allocInfo);
    }

    buffer.byteSize = desc.byteSize;

    if (createMapped)
    {
        WHEELS_ASSERT(allocInfo.pMappedData);
        buffer.mapped = allocInfo.pMappedData;
    }

    if (info.cacheDeviceAddress)
        buffer.deviceAddress =
            m_logical.getBufferAddress(vk::BufferDeviceAddressInfo{
                .buffer = buffer.handle,
            });

    m_logical.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
        .objectType = vk::ObjectType::eBuffer,
        .objectHandle =
            reinterpret_cast<uint64_t>(static_cast<VkBuffer>(buffer.handle)),
        .pObjectName = info.debugName,
    });

    if (info.initialData != nullptr)
    {
        const StaticArray postfix = "StagingBuffer";
        String stagingDebugName{
            gAllocators.general, strlen(info.debugName) + postfix.size() - 1};
        stagingDebugName.extend(info.debugName);
        stagingDebugName.extend(postfix.data());

        Buffer stagingBuffer = createBuffer(BufferCreateInfo{
            .desc =
                BufferDescription{
                    .byteSize = desc.byteSize,
                    .usage = vk::BufferUsageFlagBits::eTransferSrc,
                    .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                                  vk::MemoryPropertyFlagBits::eHostCoherent,
                },
            .debugName = stagingDebugName.c_str(),
        });
        defer { destroy(stagingBuffer); };

        WHEELS_ASSERT(stagingBuffer.mapped != nullptr);
        // Seems like a false positive
        // NOLINTNEXTLINE(clang-analyzer-nullability.NullableDereferenced)
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
    }

    trackBuffer(buffer);

    return buffer;
}

void Device::destroy(Buffer &buffer)
{
    WHEELS_ASSERT(m_initialized);

    if (buffer.handle == vk::Buffer{})
        return;

    untrackBuffer(buffer);

    auto *vkBuffer = static_cast<VkBuffer>(buffer.handle);
    {
        const std::lock_guard lock{m_allocatorMutex};
        vmaDestroyBuffer(m_allocator, vkBuffer, buffer.allocation);
    }

    buffer.handle = vk::Buffer{};
}

TexelBuffer Device::create(const TexelBufferCreateInfo &info)
{
    WHEELS_ASSERT(m_initialized);

    return createTexelBuffer(info);
}

TexelBuffer Device::createTexelBuffer(const TexelBufferCreateInfo &info)
{
    WHEELS_ASSERT(m_initialized);

    const TexelBufferDescription &desc = info.desc;
    const BufferDescription &bufferDesc = desc.bufferDesc;

    const auto formatProperties = m_physical.getFormatProperties(desc.format);

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

    const auto view = m_logical.createBufferView(vk::BufferViewCreateInfo{
        .buffer = buffer.handle,
        .format = desc.format,
        .offset = 0,
        .range = bufferDesc.byteSize,
    });

    TexelBuffer ret;
    ret.handle = buffer.handle;
    ret.view = view;
    ret.format = desc.format;
    ret.size = bufferDesc.byteSize;
    ret.allocation = buffer.allocation;

    trackTexelBuffer(ret);

    return ret;
}

void Device::destroy(TexelBuffer &buffer)
{
    WHEELS_ASSERT(m_initialized);

    if (buffer.handle == vk::Buffer{})
        return;

    untrackTexelBuffer(buffer);

    auto *vkBuffer = static_cast<VkBuffer>(buffer.handle);
    {
        const std::lock_guard lock{m_allocatorMutex};
        vmaDestroyBuffer(m_allocator, vkBuffer, buffer.allocation);
    }
    m_logical.destroy(buffer.view);

    buffer.handle = vk::Buffer{};
}

Image Device::create(const ImageCreateInfo &info)
{
    WHEELS_ASSERT(m_initialized);

    return createImage(info);
}

Image Device::createImage(const ImageCreateInfo &info)
{
    WHEELS_ASSERT(m_initialized);

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
        const std::lock_guard lock{m_allocatorMutex};
        vmaCreateImage(
            m_allocator, vkpImageInfo, &allocInfo, vkpImage, &image.allocation,
            nullptr);
    }

    m_logical.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
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
                    WHEELS_ASSERT(
                        desc.layerCount == 6 && "Cube arrays not supported");
                    return vk::ImageViewType::eCube;
                }
                return vk::ImageViewType::e2DArray;
            }
        case vk::ImageType::e3D:
            WHEELS_ASSERT(desc.layerCount == 1 && "Can't have 3D image arrays");
            return vk::ImageViewType::e3D;
        default:
            throw std::runtime_error(
                "Unexpected image type " + to_string(desc.imageType));
        }
    }();

    image.view = m_logical.createImageView(vk::ImageViewCreateInfo{
        .image = image.handle,
        .viewType = viewType,
        .format = desc.format,
        .subresourceRange = range,
    });

    m_logical.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
        .objectType = vk::ObjectType::eImageView,
        .objectHandle =
            reinterpret_cast<uint64_t>(static_cast<VkImageView>(image.view)),
        .pObjectName = info.debugName,
    });

    image.extent = extent;
    image.mipCount = desc.mipCount;
    image.subresourceRange = range;
    image.imageType = desc.imageType;
    image.format = desc.format;

    {
        VmaAllocationInfo vmaInfo;
        {
            const std::lock_guard lock{m_allocatorMutex};
            vmaGetAllocationInfo(m_allocator, image.allocation, &vmaInfo);
        }
        image.rawByteSize = vmaInfo.size;
    }

    trackImage(image);

    return image;
}

void Device::destroy(Image &image)
{
    WHEELS_ASSERT(m_initialized);

    if (image.handle == vk::Image{})
        return;

    untrackImage(image);

    auto *vkImage = static_cast<VkImage>(image.handle);
    {
        const std::lock_guard lock{m_allocatorMutex};
        vmaDestroyImage(m_allocator, vkImage, image.allocation);
    }
    m_logical.destroy(image.view);

    image.handle = vk::Image{};
}

void Device::createSubresourcesViews(
    const Image &image, StrSpan debugNamePrefix,
    Span<vk::ImageView> outViews) const
{
    WHEELS_ASSERT(m_initialized);

    WHEELS_ASSERT(
        image.subresourceRange.layerCount == 1 &&
        "Texture arrays not supported");
    WHEELS_ASSERT(
        image.subresourceRange.levelCount > 1 &&
        "You can just use the global view when no mips are present");
    WHEELS_ASSERT(image.subresourceRange.baseMipLevel == 0);
    WHEELS_ASSERT(image.subresourceRange.levelCount == outViews.size());

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

    String tmp{gAllocators.general};
    tmp.reserve(debugNamePrefix.size() + 2);
    tmp.extend(debugNamePrefix);
    for (uint32_t i = 0; i < image.subresourceRange.levelCount; ++i)
    {
        outViews[i] = m_logical.createImageView(vk::ImageViewCreateInfo{
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

        // We won't have three digits for mip level as each mip halves the size
        // Do math in int32_t to account for implementation defined signedness
        // for char
        if (i >= 10)
            tmp.push_back(asserted_cast<char>(
                asserted_cast<int32_t>('0') + asserted_cast<int32_t>(i) / 10));
        tmp.push_back(asserted_cast<char>(
            asserted_cast<int32_t>('0') + asserted_cast<int32_t>(i) % 10));

        m_logical.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
            .objectType = vk::ObjectType::eImageView,
            .objectHandle = reinterpret_cast<uint64_t>(
                static_cast<VkImageView>(image.view)),
            .pObjectName = tmp.c_str(),
        });

        if (i >= 10)
            tmp.pop_back();
        tmp.pop_back();
    }
}

void Device::destroy(Span<const vk::ImageView> views) const
{
    WHEELS_ASSERT(m_initialized);

    for (const vk::ImageView view : views)
        m_logical.destroy(view);
}

vk::CommandBuffer Device::beginGraphicsCommands() const
{
    WHEELS_ASSERT(m_initialized);

    const auto buffer =
        m_logical.allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = m_graphicsPool,
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
    WHEELS_ASSERT(m_initialized);

    buffer.end();

    const vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &buffer,
    };
    checkSuccess(m_graphicsQueue.submit(1, &submitInfo, vk::Fence{}), "submit");
    m_graphicsQueue.waitIdle();

    m_logical.freeCommandBuffers(m_graphicsPool, 1, &buffer);
}

const MemoryAllocationBytes &Device::memoryAllocations() const
{
    WHEELS_ASSERT(m_initialized);

    return m_memoryAllocations;
}

bool Device::isDeviceSuitable(
    ScopedScratch scopeAlloc, const vk::PhysicalDevice device) const
{
    const auto families = findQueueFamilies(device, m_surface);
    if (!families.isComplete())
    {
        LOG_ERR("Missing required queue families");
        return false;
    }

    if (!checkDeviceExtensionSupport(
            scopeAlloc.child_scope(), device, m_settings))
        return false;

    const SwapchainSupport swapSupport{scopeAlloc, device, m_surface};
    if (swapSupport.formats.empty() || swapSupport.presentModes.empty())
    {
        LOG_ERR("Inadequate swap chain");
        return false;
    }

    LOG_INFO("Checking feature support");

    {
        const auto features = device.getFeatures2<ALL_FEATURE_STRUCTS_LIST>();

#define CHECK_REQUIRED_FEATURES(container, feature)                            \
    if (features.get<container>().feature == VK_FALSE)                         \
    {                                                                          \
        LOG_ERR("Missing %s", #feature);                                       \
        return false;                                                          \
    }

        FOR_EACH_PAIR(CHECK_REQUIRED_FEATURES, REQUIRED_FEATURES);

#undef CHECK_REQUIRED_FEATURES
    }

    if (m_settings.robustAccess)
    {
        const auto allFeatures = device.getFeatures2<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceRobustness2FeaturesEXT>();

        const vk::PhysicalDeviceFeatures2 &features =
            allFeatures.get<vk::PhysicalDeviceFeatures2>();

        // robustBufferAccess2 requires enabling robustBufferAccess
        if (features.features.robustBufferAccess == VK_FALSE)
        {
            LOG_ERR("Missing robustBufferAccess");
            return false;
        }

        const vk::PhysicalDeviceRobustness2FeaturesEXT &robustness =
            allFeatures.get<vk::PhysicalDeviceRobustness2FeaturesEXT>();
        if (robustness.robustBufferAccess2 == VK_FALSE)
        {
            LOG_ERR("Missing robustBufferAccess2");
            return false;
        }
        if (robustness.robustImageAccess2 == VK_FALSE)
        {
            LOG_ERR("Missing robustImageAccess2");
            return false;
        }
    }

    const auto props = device.getProperties2<
        vk::PhysicalDeviceProperties2, vk::PhysicalDeviceSubgroupProperties>();

    {
        const vk::PhysicalDeviceProperties2 &deviceProps =
            props.get<vk::PhysicalDeviceProperties2>();

        if (deviceProps.properties.apiVersion < VK_VERSION_1_3)
        {
            LOG_ERR("Missing Vulkan 1.3 support");
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
            LOG_ERR("Missing subgroup arithmetic op support");
            return false;
        }
    }

    LOG_INFO("Required features are supported");

    return true;
}

void Device::createInstance(ScopedScratch scopeAlloc)
{
    if (m_settings.enableDebugLayers && !checkValidationLayerSupport())
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

    m_instance = vk::createInstance(vk::InstanceCreateInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount =
            m_settings.enableDebugLayers
                ? asserted_cast<uint32_t>(validationLayers.size())
                : 0,
        .ppEnabledLayerNames =
            m_settings.enableDebugLayers ? validationLayers.data() : nullptr,
        .enabledExtensionCount =
            asserted_cast<uint32_t>(extension_cstrs.size()),
        .ppEnabledExtensionNames = extension_cstrs.data(),
    });
}

void Device::createDebugMessenger()
{
    const vk::DebugUtilsMessengerCreateInfoEXT createInfo{
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation,
        .pfnUserCallback = debugCallback,
        .pUserData = &m_settings,
    };
    CreateDebugUtilsMessengerEXT(
        m_instance, &createInfo, nullptr, &m_debugMessenger);
}

void Device::createSurface()
{
    GLFWwindow *window = gWindow.ptr();
    WHEELS_ASSERT(window != nullptr);

    auto *vkpSurface = reinterpret_cast<VkSurfaceKHR *>(&m_surface);
    auto *vkInstance = static_cast<VkInstance>(m_instance);
    if (glfwCreateWindowSurface(vkInstance, window, nullptr, vkpSurface) !=
        VK_SUCCESS)
        throw std::runtime_error("Failed to create window surface");
}

void Device::selectPhysicalDevice(ScopedScratch scopeAlloc)
{
    LOG_INFO("Selecting device");

    const auto devices = m_instance.enumeratePhysicalDevices();

    for (const auto &device : devices)
    {
        LOG_INFO("Considering '%s'", device.getProperties().deviceName.data());
        if (isDeviceSuitable(scopeAlloc.child_scope(), device))
        {
            m_physical = device;
            return;
        }
    }

    throw std::runtime_error("Failed to find a suitable GPU");
}

void Device::createLogicalDevice(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(m_queueFamilies.graphicsFamily.has_value());
    WHEELS_ASSERT(m_queueFamilies.transferFamily.has_value());

    const uint32_t graphicsFamily = *m_queueFamilies.graphicsFamily;
    const uint32_t graphicsFamilyQueueCount =
        m_queueFamilies.graphicsFamilyQueueCount;
    const uint32_t transferFamily = *m_queueFamilies.transferFamily;

    // First queue in family has largest queue, rest descend
    const StaticArray queuePriorities{{1.f, 0.f}};
    const InlineArray<vk::DeviceQueueCreateInfo, 2> queueCreateInfos = [&]
    {
        InlineArray<vk::DeviceQueueCreateInfo, 2> cis;
        if (graphicsFamily == transferFamily)
        {
            WHEELS_ASSERT(queuePriorities.size() >= 2);
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

    Array<const char *> enabledExtensions{
        scopeAlloc, deviceExtensions.size() + 1};
    for (const char *ext : deviceExtensions)
        enabledExtensions.push_back(ext);

    const char *robustness2Name = VK_EXT_ROBUSTNESS_2_EXTENSION_NAME;
    if (m_settings.robustAccess)
        enabledExtensions.push_back(robustness2Name);

    vk::StructureChain<
        vk::DeviceCreateInfo, ALL_FEATURE_STRUCTS_LIST,
        vk::PhysicalDeviceRobustness2FeaturesEXT>
        createChain;
    createChain.get<vk::DeviceCreateInfo>() = vk::DeviceCreateInfo{
        .pNext = &createChain.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount =
            asserted_cast<uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledLayerCount =
            m_settings.enableDebugLayers
                ? asserted_cast<uint32_t>(validationLayers.size())
                : 0,
        .ppEnabledLayerNames =
            m_settings.enableDebugLayers ? validationLayers.data() : nullptr,
        .enabledExtensionCount =
            asserted_cast<uint32_t>(enabledExtensions.size()),
        .ppEnabledExtensionNames = enabledExtensions.data(),
    };

#define TOGGLE_REQUIRED_FEATURES(container, feature)                           \
    createChain.get<container>().feature = VK_TRUE;

    FOR_EACH_PAIR(TOGGLE_REQUIRED_FEATURES, REQUIRED_FEATURES);

#undef TOGGLE_REQUIRED_FEATURES

    if (m_settings.robustAccess)
    {
        createChain.get<vk::PhysicalDeviceFeatures2>()
            .features.robustBufferAccess = VK_TRUE;
        createChain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>() =
            vk::PhysicalDeviceRobustness2FeaturesEXT{
                .robustBufferAccess2 = VK_TRUE,
                .robustImageAccess2 = VK_TRUE,
            };
    }
    else
        createChain.unlink<vk::PhysicalDeviceRobustness2FeaturesEXT>();

    m_logical =
        m_physical.createDevice(createChain.get<vk::DeviceCreateInfo>());

    m_graphicsQueue = m_logical.getQueue(graphicsFamily, 0);
    if (graphicsFamily == transferFamily)
    {
        WHEELS_ASSERT(
            graphicsFamilyQueueCount > 1 &&
            "Device doesn't support two queues");
        m_transferQueue = m_logical.getQueue(graphicsFamily, 1);
    }
    else
        m_transferQueue = m_logical.getQueue(transferFamily, 0);
}

void Device::createAllocator()
{
    const VmaAllocatorCreateInfo allocatorInfo{
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = static_cast<VkPhysicalDevice>(m_physical),
        .device = static_cast<VkDevice>(m_logical),
        .instance = static_cast<VkInstance>(m_instance),
    };
    if (vmaCreateAllocator(&allocatorInfo, &m_allocator) != VK_SUCCESS)
        throw std::runtime_error("Failed to create allocator");
}

void Device::createCommandPools()
{
    {
        WHEELS_ASSERT(m_queueFamilies.graphicsFamily.has_value());

        const vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = *m_queueFamilies.graphicsFamily,
        };
        m_graphicsPool = m_logical.createCommandPool(poolInfo, nullptr);
    }
    {
        WHEELS_ASSERT(m_queueFamilies.transferFamily.has_value());

        const vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = *m_queueFamilies.transferFamily,
        };
        m_transferPool = m_logical.createCommandPool(poolInfo, nullptr);
    }
}

void Device::trackBuffer(const Buffer &buffer)
{
    VmaAllocationInfo info;
    {
        const std::lock_guard lock{m_allocatorMutex};
        vmaGetAllocationInfo(m_allocator, buffer.allocation, &info);
    }

    m_memoryAllocations.buffers += info.size;
}

void Device::untrackBuffer(const Buffer &buffer)
{
    if (buffer.allocation == nullptr)
        return;

    VmaAllocationInfo info;
    {
        const std::lock_guard lock{m_allocatorMutex};
        vmaGetAllocationInfo(m_allocator, buffer.allocation, &info);
    }

    m_memoryAllocations.buffers -= info.size;
}

void Device::trackTexelBuffer(const TexelBuffer &buffer)
{
    VmaAllocationInfo info;
    {
        const std::lock_guard lock{m_allocatorMutex};
        vmaGetAllocationInfo(m_allocator, buffer.allocation, &info);
    }

    m_memoryAllocations.texelBuffers += info.size;
}

void Device::untrackTexelBuffer(const TexelBuffer &buffer)
{
    if (buffer.allocation == nullptr)
        return;

    VmaAllocationInfo info;
    {
        const std::lock_guard lock{m_allocatorMutex};
        vmaGetAllocationInfo(m_allocator, buffer.allocation, &info);
    }

    m_memoryAllocations.texelBuffers -= info.size;
}

void Device::trackImage(const Image &image)
{
    VmaAllocationInfo info;
    {
        const std::lock_guard lock{m_allocatorMutex};
        vmaGetAllocationInfo(m_allocator, image.allocation, &info);
    }

    m_memoryAllocations.images += info.size;
}

void Device::untrackImage(const Image &image)
{
    if (image.allocation == nullptr)
        return;

    VmaAllocationInfo info;
    {
        const std::lock_guard lock{m_allocatorMutex};
        vmaGetAllocationInfo(m_allocator, image.allocation, &info);
    }

    m_memoryAllocations.images -= info.size;
}

std::filesystem::path Device::updateShaderCache(
    Allocator &alloc, const std::filesystem::path &sourcePath,
    StrSpan topLevelSource, const std::filesystem::path &relPath)
{
    HashSet<std::filesystem::path> uniqueIncludes{alloc};
    // Also push root file as reflection expects all sources to be included here
    uniqueIncludes.insert(sourcePath.lexically_normal());

    String fullSource{alloc};
    try
    {
        expandIncludes(
            alloc, sourcePath, topLevelSource, fullSource, uniqueIncludes, 0);
    }
    catch (const std::exception &e)
    {
        // Just log so that the calling code can skip without error on recompile
        LOG_ERR("%s", e.what());
        return {};
    }

    // wyhash should be fine here, it's effectively 62bit for collisions
    // https://github.com/Cyan4973/xxHash/issues/236#issuecomment-522051621
    const uint64_t sourceHash =
        wyhash(fullSource.data(), fullSource.size(), 0, (uint64_t const *)_wyp);
    StaticArray<char, sizeof(uint64_t) * 2 + 1> hashStr;
    snprintf(hashStr.data(), hashStr.size(), "%" PRIX64, sourceHash);

    std::filesystem::path cachePath =
        resPath(std::filesystem::path("shader") / "cache" / hashStr.data());
    cachePath.replace_extension("prosper_shader");

    const bool cacheValid = readCache(alloc, cachePath);
    if (!cacheValid || m_settings.dumpShaderDisassembly)
    {
        LOG_INFO("Compiling %s", relPath.string().c_str());

        const shaderc::SpvCompilationResult result =
            m_compiler->CompileGlslToSpv(
                fullSource.c_str(), fullSource.size(),
                shaderc_glsl_infer_from_source, sourcePath.string().c_str(),
                m_compilerOptions);

        if (const auto status = result.GetCompilationStatus(); status)
        {
            const auto err = result.GetErrorMessage();
            if (err.empty())
                LOG_ERR(
                    "Compilation of '%s' failed\n%s",
                    sourcePath.string().c_str(), statusString(status));
            else
                LOG_ERR(
                    "Compilation of '%s' failed\n%s\n%s",
                    sourcePath.string().c_str(), statusString(status),
                    err.c_str());
            return {};
        }

        writeCache(cachePath, result, uniqueIncludes);

        if (m_settings.dumpShaderDisassembly)
        {
            const shaderc::AssemblyCompilationResult resultAsm =
                m_compiler->CompileGlslToSpvAssembly(
                    fullSource.c_str(), fullSource.size(),
                    shaderc_glsl_infer_from_source, sourcePath.string().c_str(),
                    m_compilerOptions);
            if (const shaderc_compilation_status status =
                    result.GetCompilationStatus();
                status == shaderc_compilation_status_success)
                LOG_INFO("%s", resultAsm.begin());
            else
            {
                const std::string err = result.GetErrorMessage();
                if (err.empty())
                    LOG_ERR(
                        "Compilation of '%s' failed\n%s",
                        sourcePath.string().c_str(), statusString(status));
                else
                    LOG_ERR(
                        "Compilation of '%s' failed\n%s\n%s",
                        sourcePath.string().c_str(), statusString(status),
                        err.c_str());
                return {};
            }
        }
    }
    else
        LOG_INFO("Loading '%s' from cache", relPath.string().c_str());

    return cachePath;
}
