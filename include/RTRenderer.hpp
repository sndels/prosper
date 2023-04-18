#ifndef PROSPER_RT_RENDERER_HPP
#define PROSPER_RT_RENDERER_HPP

#include "Camera.hpp"
#include "DebugDrawTypes.hpp"
#include "Device.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "World.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

class RTRenderer
{
  public:
    enum class DrawType : uint32_t
    {
        Default = 0,
        DEBUG_DRAW_TYPES_AND_COUNT
    };

    RTRenderer(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, const SwapchainConfig &swapConfig,
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    ~RTRenderer();

    RTRenderer(const RTRenderer &other) = delete;
    RTRenderer(RTRenderer &&other) = delete;
    RTRenderer &operator=(const RTRenderer &other) = delete;
    RTRenderer &operator=(RTRenderer &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    void recreate(
        wheels::ScopedScratch scopeAlloc, const SwapchainConfig &swapConfig,
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    void drawUi();

    void record(
        vk::CommandBuffer cb, const World &world, const Camera &cam,
        const vk::Rect2D &renderArea, uint32_t nextImage,
        Profiler *profiler) const;

  private:
    void destroyShaders();
    void destroySwapchainRelated();
    void destroyPipeline();

    [[nodiscard]] bool compileShaders(
        wheels::ScopedScratch scopeAlloc,
        const World::DSLayouts &worldDSLayouts);

    // These also need to be recreated with Swapchain as they depend on
    // swapconfig
    void createDescriptorSets(const SwapchainConfig &swapConfig);
    void createPipeline(
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    void createShaderBindingTable(wheels::ScopedScratch scopeAlloc);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 3> _shaderStages{{}};
    wheels::StaticArray<vk::RayTracingShaderGroupCreateInfoKHR, 3>
        _shaderGroups{{}};

    vk::DescriptorSetLayout _descriptorSetLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_SWAPCHAIN_IMAGES>
        _descriptorSets;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;

    vk::DeviceSize _sbtGroupSize{0};
    Buffer _shaderBindingTable;

    DrawType _drawType{DrawType::Default};
};

#endif // PROSPER_RT_RENDERER_HPP
