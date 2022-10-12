#ifndef PROSPER_DEVICE_HPP
#define PROSPER_DEVICE_HPP

#include "Resources.hpp"

#include <GLFW/glfw3.h>
#include <shaderc/shaderc.hpp>

#include <filesystem>
#include <optional>
#include <unordered_set>
#include <vector>

struct QueueFamilies
{
    std::optional<uint32_t> computeFamily;
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    [[nodiscard]] bool isComplete() const
    {
        return computeFamily.has_value() && graphicsFamily.has_value() &&
               presentFamily.has_value();
    }
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
    FileIncluder();

    shaderc_include_result *GetInclude(
        const char *requested_source, shaderc_include_type type,
        const char *requesting_source, size_t include_depth) override;

    void ReleaseInclude(shaderc_include_result *data) override;

  private:
    std::filesystem::path _includePath;

    uint64_t _includeContentID{0};
    std::unordered_map<
        uint64_t,
        std::tuple<std::shared_ptr<shaderc_include_result>, std::string>>
        _includeContent;
};

class Device
{
  public:
    Device(GLFWwindow *window, bool enableDebugLayers);
    ~Device();

    Device(const Device &other) = delete;
    Device(Device &&other) = delete;
    Device &operator=(const Device &other) = delete;
    Device &operator=(Device &&other) = delete;

    [[nodiscard]] vk::Instance instance() const;
    [[nodiscard]] vk::PhysicalDevice physical() const;
    [[nodiscard]] vk::Device logical() const;
    [[nodiscard]] vk::SurfaceKHR surface() const;
    [[nodiscard]] vk::CommandPool computePool() const;
    [[nodiscard]] vk::CommandPool graphicsPool() const;
    [[nodiscard]] vk::Queue computeQueue() const;
    [[nodiscard]] vk::Queue graphicsQueue() const;
    [[nodiscard]] vk::Queue presentQueue() const;
    [[nodiscard]] const QueueFamilies &queueFamilies() const;
    [[nodiscard]] const DeviceProperties &properties() const;

    struct CompileShaderModuleArgs
    {
        const std::string &relPath;
        const std::string &debugName;
        // Default value lives for the duration of the object
        const std::string &defines{""};
    };
    [[nodiscard]] std::optional<vk::ShaderModule> compileShaderModule(
        const CompileShaderModuleArgs &info) const;

    [[nodiscard]] void *map(Buffer const &buffer) const;
    void unmap(Buffer const &buffer) const;
    [[nodiscard]] void *map(Image const &texture) const;
    void unmap(Image const &texture) const;

    [[nodiscard]] Buffer createBuffer(const BufferCreateInfo &info) const;
    void destroy(const Buffer &buffer) const;

    [[nodiscard]] TexelBuffer createTexelBuffer(
        const TexelBufferCreateInfo &info) const;
    void destroy(const TexelBuffer &buffer) const;

    [[nodiscard]] Image createImage(const ImageCreateInfo &info) const;
    void destroy(const Image &image) const;

    [[nodiscard]] vk::CommandBuffer beginGraphicsCommands() const;
    void endGraphicsCommands(vk::CommandBuffer buffer) const;

  private:
    [[nodiscard]] bool isDeviceSuitable(vk::PhysicalDevice device) const;

    [[nodiscard]] void *map(VmaAllocation allocation) const;
    void unmap(VmaAllocation allocation) const;

    void createInstance(bool enableDebugLayers);
    void createDebugMessenger();
    void createSurface(GLFWwindow *window);
    void selectPhysicalDevice();
    void createLogicalDevice(bool enableDebugLayers);
    void createAllocator();
    void createCommandPools();

    vk::Instance _instance;
    vk::PhysicalDevice _physical;
    vk::Device _logical;
    DeviceProperties _properties;

    VmaAllocator _allocator{nullptr};

    shaderc::CompileOptions _compilerOptions;
    shaderc::Compiler _compiler;

    vk::SurfaceKHR _surface;

    QueueFamilies _queueFamilies;
    vk::Queue _computeQueue;
    vk::Queue _graphicsQueue;
    vk::Queue _presentQueue;

    vk::CommandPool _computePool;
    vk::CommandPool _graphicsPool;

    vk::DebugUtilsMessengerEXT _debugMessenger;
};

#endif // PROSPER_DEVICE_HPP
