#ifndef PROSPER_DEVICE_HPP
#define PROSPER_DEVICE_HPP

#include "Resources.hpp"
#include "ShaderReflection.hpp"

#include <GLFW/glfw3.h>
#include <shaderc/shaderc.hpp>

#include <wheels/allocators/cstdlib_allocator.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/hash_map.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/string.hpp>

#include <filesystem>

struct QueueFamilies
{
    wheels::Optional<uint32_t> graphicsFamily;

    [[nodiscard]] bool isComplete() const { return graphicsFamily.has_value(); }
};

struct DeviceProperties
{
    vk::PhysicalDeviceProperties device;
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rtPipeline;
    vk::PhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructure;
};

class FileIncluder : public shaderc::CompileOptions::IncluderInterface
{
  public:
    FileIncluder(wheels::Allocator &alloc);

    shaderc_include_result *GetInclude(
        const char *requested_source, shaderc_include_type type,
        const char *requesting_source, size_t include_depth) override;

    void ReleaseInclude(shaderc_include_result *data) override;

  private:
    wheels::Allocator &_alloc;

    std::filesystem::path _includePath;

    uint64_t _includeContentID{0};
    struct IncludeContent
    {
        std::unique_ptr<shaderc_include_result> result{nullptr};
        std::unique_ptr<wheels::String> content{nullptr};
        std::unique_ptr<wheels::String> path{nullptr};
    };
    wheels::HashMap<uint64_t, IncludeContent> _includeContent;
};

struct MemoryAllocationBytes
{
    vk::DeviceSize images{0};
    vk::DeviceSize buffers{0};
    vk::DeviceSize texelBuffers{0};
};

class Device
{
  public:
    struct Settings
    {
        bool enableDebugLayers{false};
        bool dumpShaderDisassembly{false};
    };

    Device(
        wheels::ScopedScratch scopeAlloc, GLFWwindow *window,
        const Settings &settings);
    ~Device();

    Device(const Device &other) = delete;
    Device(Device &&other) = delete;
    Device &operator=(const Device &other) = delete;
    Device &operator=(Device &&other) = delete;

    [[nodiscard]] vk::Instance instance() const;
    [[nodiscard]] vk::PhysicalDevice physical() const;
    [[nodiscard]] vk::Device logical() const;
    [[nodiscard]] vk::SurfaceKHR surface() const;
    [[nodiscard]] vk::CommandPool graphicsPool() const;
    [[nodiscard]] vk::Queue graphicsQueue() const;
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
    [[nodiscard]] wheels::Optional<ShaderCompileResult> compileShaderModule(
        wheels::ScopedScratch scopeAlloc, const CompileShaderModuleArgs &info);

    // TODO: Should this take in an allocator for the reflection and
    // not use the interal general one?
    [[nodiscard]] wheels::Optional<ShaderReflection> reflectShader(
        wheels::ScopedScratch scopeAlloc, const CompileShaderModuleArgs &info,
        bool add_dummy_compute_boilerplate);

    [[nodiscard]] Buffer create(const BufferCreateInfo &info);
    [[nodiscard]] Buffer createBuffer(const BufferCreateInfo &info);
    void destroy(const Buffer &buffer);

    [[nodiscard]] TexelBuffer create(const TexelBufferCreateInfo &info);
    [[nodiscard]] TexelBuffer createTexelBuffer(
        const TexelBufferCreateInfo &info);
    void destroy(const TexelBuffer &buffer);

    [[nodiscard]] Image create(const ImageCreateInfo &info);
    [[nodiscard]] Image createImage(const ImageCreateInfo &info);
    void destroy(const Image &image);

    [[nodiscard]] vk::CommandBuffer beginGraphicsCommands() const;
    void endGraphicsCommands(vk::CommandBuffer buffer) const;

    [[nodiscard]] const MemoryAllocationBytes &memoryAllocations() const;

  private:
    [[nodiscard]] bool isDeviceSuitable(
        wheels::ScopedScratch scopeAlloc, vk::PhysicalDevice device) const;

    [[nodiscard]] void *map(VmaAllocation allocation) const;
    void unmap(VmaAllocation allocation) const;

    void createInstance(
        wheels::ScopedScratch scopeAlloc, bool enableDebugLayers);
    void createDebugMessenger();
    void createSurface(GLFWwindow *window);
    void selectPhysicalDevice(wheels::ScopedScratch scopeAlloc);
    void createLogicalDevice(
        wheels::ScopedScratch scopeAlloc, bool enableDebugLayers);
    void createAllocator();
    void createCommandPools();

    void trackBuffer(const Buffer &buffer);
    void untrackBuffer(const Buffer &buffer);
    void trackTexelBuffer(const TexelBuffer &buffer);
    void untrackTexelBuffer(const TexelBuffer &buffer);
    void trackImage(const Image &image);
    void untrackImage(const Image &image);

    wheels::CstdlibAllocator _generalAlloc;

    vk::Instance _instance;
    vk::PhysicalDevice _physical;
    vk::Device _logical;
    DeviceProperties _properties;

    VmaAllocator _allocator{nullptr};

    bool _dumpShaderDisassembly{false};
    shaderc::CompileOptions _compilerOptions;
    shaderc::Compiler _compiler;

    vk::SurfaceKHR _surface;

    QueueFamilies _queueFamilies;
    vk::Queue _graphicsQueue;

    vk::CommandPool _graphicsPool;

    vk::DebugUtilsMessengerEXT _debugMessenger;

    MemoryAllocationBytes _memoryAllocations;
};

#endif // PROSPER_DEVICE_HPP
