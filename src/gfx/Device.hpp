#ifndef PROSPER_GFX_DEVICE_HPP
#define PROSPER_GFX_DEVICE_HPP

#include "gfx/Resources.hpp"
#include "gfx/ShaderReflection.hpp"
#include "utils/Hashes.hpp"

#include <atomic>
#include <filesystem>
#include <mutex>
#include <shaderc/shaderc.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/hash_set.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/owning_ptr.hpp>

struct QueueFamilies
{
    wheels::Optional<uint32_t> graphicsFamily;
    uint32_t graphicsFamilyQueueCount{0};
    wheels::Optional<uint32_t> computeFamily;
    uint32_t computeFamilyQueueCount{0};
    wheels::Optional<uint32_t> transferFamily;
    uint32_t transferFamilyQueueCount{0};

    [[nodiscard]] bool isComplete() const
    {
        return graphicsFamily.has_value() && graphicsFamilyQueueCount > 0 &&
               computeFamily.has_value() && computeFamilyQueueCount > 0 &&
               transferFamily.has_value() && transferFamilyQueueCount > 0;
    }
};

struct DeviceProperties
{
    vk::PhysicalDeviceProperties device;
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rtPipeline;
    vk::PhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructure;
    vk::PhysicalDeviceMeshShaderPropertiesEXT meshShader;
    vk::PhysicalDeviceSubgroupProperties subgroup;
};

struct MemoryAllocationBytes
{
    std::atomic<vk::DeviceSize> images{0};
    std::atomic<vk::DeviceSize> buffers{0};
    std::atomic<vk::DeviceSize> texelBuffers{0};
};

// Interfaces not labelled thread-unsafe can be assumed to be thread safe.
// TODO: Checks for races, UnnecessaryLock from Gregory or something
class Device
{
  public:
    struct Settings
    {
        bool enableDebugLayers{false};
        bool dumpShaderDisassembly{false};
        bool breakOnValidationError{false};
        bool robustAccess{false};
    };

    Device() noexcept = default;
    ~Device();

    Device(const Device &other) = delete;
    Device(Device &&other) = delete;
    Device &operator=(const Device &other) = delete;
    Device &operator=(Device &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc, const Settings &settings);
    void destroy();

    [[nodiscard]] vk::Instance instance() const;
    [[nodiscard]] vk::PhysicalDevice physical() const;
    [[nodiscard]] vk::Device logical() const;
    [[nodiscard]] vk::SurfaceKHR surface() const;
    [[nodiscard]] vk::CommandPool graphicsPool() const;
    [[nodiscard]] vk::Queue graphicsQueue() const;
    [[nodiscard]] vk::CommandPool transferPool() const;
    [[nodiscard]] vk::Queue transferQueue() const;
    [[nodiscard]] const QueueFamilies &queueFamilies() const;
    [[nodiscard]] const DeviceProperties &properties() const;

    struct CompileShaderModuleArgs
    {
        // Potential temporaries (and default values) referred to live for the
        // duration of the object as binding to the const lvalue ref extends
        // the lifetimes of temporaries to match the object.
        // Note that the full extension requires {}-initialization and doesn't
        // work if 'new' is involved.
        // Don't know if this is a neat way to do named args or a footgun.
        const std::filesystem::path &relPath;
        const char *debugName{nullptr};
        wheels::StrSpan defines{""};
    };
    struct ShaderCompileResult
    {
        vk::ShaderModule module;
        ShaderReflection reflection;
    };
    // TODO: Should this take in an allocator for the reflection and
    // not use the interal general one?
    // This is not thread-safe
    [[nodiscard]] wheels::Optional<ShaderCompileResult> compileShaderModule(
        wheels::ScopedScratch scopeAlloc, const CompileShaderModuleArgs &info);

    // TODO: Should this take in an allocator for the reflection and
    // not use the interal general one?
    // This is not thread-safe
    [[nodiscard]] wheels::Optional<ShaderReflection> reflectShader(
        wheels::ScopedScratch scopeAlloc, const CompileShaderModuleArgs &info,
        bool add_dummy_compute_boilerplate);

    // Initial data can only be given if the thread has exclusive access to
    // graphicsPool and graphicsQueue.
    [[nodiscard]] Buffer create(const BufferCreateInfo &info);
    // Initial data can only be given if the thread has exclusive access to
    // graphicsPool and graphicsQueue.
    [[nodiscard]] Buffer createBuffer(const BufferCreateInfo &info);
    // buffer shouldn't be in use in other threads
    void destroy(Buffer &buffer);

    // Initial data can only be given if the thread has exclusive access to
    // graphicsPool and graphicsQueue.
    [[nodiscard]] TexelBuffer create(const TexelBufferCreateInfo &info);
    // Initial data can only be given if the thread has exclusive access to
    // graphicsPool and graphicsQueue.
    [[nodiscard]] TexelBuffer createTexelBuffer(
        const TexelBufferCreateInfo &info);
    // buffer shouldn't be in use in other threads
    void destroy(TexelBuffer &buffer);

    [[nodiscard]] Image create(const ImageCreateInfo &info);
    [[nodiscard]] Image createImage(const ImageCreateInfo &info);
    // image shouldn't be in use in other threads
    void destroy(Image &image);
    // Creates views to the individual subresources of the image
    // NOTE: Caller is responsible of destroying the views
    void createSubresourcesViews(
        const Image &image, wheels::StrSpan debugNamePrefix,
        wheels::Span<vk::ImageView> outViews) const;
    void destroy(wheels::Span<const vk::ImageView> views) const;

    // This is not thread-safe
    [[nodiscard]] vk::CommandBuffer beginGraphicsCommands() const;
    // This is not thread-safe
    void endGraphicsCommands(vk::CommandBuffer buffer) const;

    [[nodiscard]] const MemoryAllocationBytes &memoryAllocations() const;

  private:
    [[nodiscard]] bool isDeviceSuitable(
        wheels::ScopedScratch scopeAlloc, vk::PhysicalDevice device) const;

    void createInstance(wheels::ScopedScratch scopeAlloc);
    void createDebugMessenger();
    void createSurface();
    void selectPhysicalDevice(wheels::ScopedScratch scopeAlloc);
    void createLogicalDevice(wheels::ScopedScratch scopeAlloc);
    void createAllocator();
    void createCommandPools();

    void trackBuffer(const Buffer &buffer);
    void untrackBuffer(const Buffer &buffer);
    void trackTexelBuffer(const TexelBuffer &buffer);
    void untrackTexelBuffer(const TexelBuffer &buffer);
    void trackImage(const Image &image);
    void untrackImage(const Image &image);

    std::filesystem::path updateShaderCache(
        wheels::Allocator &alloc, const std::filesystem::path &sourcePath,
        wheels::StrSpan topLevelSource, const std::filesystem::path &relPath);

    // All members should init (in ctor) without dynamic allocations or
    // exceptions because this class is used in a extern global.

    bool m_initialized{false};
    Settings m_settings;

    vk::Instance m_instance;
    vk::PhysicalDevice m_physical;
    vk::Device m_logical;
    DeviceProperties m_properties;

    std::mutex m_allocatorMutex;
    VmaAllocator m_allocator{nullptr};

    shaderc::CompileOptions m_compilerOptions;
    // Put behind a ptr to control lifetime and make the leaky mutex visible in
    // win crt debugging.
    wheels::OwningPtr<shaderc::Compiler> m_compiler;

    vk::SurfaceKHR m_surface;

    QueueFamilies m_queueFamilies;
    vk::Queue m_graphicsQueue;
    vk::Queue m_transferQueue;

    vk::CommandPool m_graphicsPool;
    vk::CommandPool m_transferPool;

    vk::DebugUtilsMessengerEXT m_debugMessenger;

    MemoryAllocationBytes m_memoryAllocations;
};

// This is depended on by Device and init()/destroy() order relative to other
// similar globals is handled in main()
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern Device gDevice;

#endif // PROSPER_GFX_DEVICE_HPP
