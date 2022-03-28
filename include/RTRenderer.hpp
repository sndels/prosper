#ifndef PROSPER_RT_RENDERER_HPP
#define PROSPER_RT_RENDERER_HPP

#include <functional>

#include "Camera.hpp"
#include "Device.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "World.hpp"

class RTRenderer
{
  public:
    RTRenderer(
        Device *device, RenderResources *resources,
        const SwapchainConfig &swapConfig, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    ~RTRenderer();

    RTRenderer(const RTRenderer &other) = delete;
    RTRenderer(RTRenderer &&other) = delete;
    RTRenderer &operator=(const RTRenderer &other) = delete;
    RTRenderer &operator=(RTRenderer &&other) = delete;

    void recompileShaders(
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    void recreateSwapchainRelated(
        const SwapchainConfig &swapConfig, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    [[nodiscard]] vk::CommandBuffer recordCommandBuffer(
        const World &world, const Camera &cam, const vk::Rect2D &renderArea,
        uint32_t nextImage) const;

  private:
    void destroyShaders();
    void destroySwapchainRelated();
    void destroyPipeline();

    [[nodiscard]] bool compileShaders();

    // These also need to be recreated with Swapchain as they depend on
    // swapconfig
    void createDescriptorSets(const SwapchainConfig &swapConfig);
    void createPipeline(
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    void createShaderBindingTable();
    void createCommandBuffers(const SwapchainConfig &swapConfig);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    std::array<vk::PipelineShaderStageCreateInfo, 3> _shaderStages;
    std::array<vk::RayTracingShaderGroupCreateInfoKHR, 3> _shaderGroups;

    vk::DescriptorSetLayout _descriptorSetLayout;
    std::vector<vk::DescriptorSet> _descriptorSets;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;

    uint32_t _sbtGroupSize{0};
    Buffer _shaderBindingTable;

    std::vector<vk::CommandBuffer> _commandBuffers;
};

#endif // PROSPER_RT_RENDERER_HPP