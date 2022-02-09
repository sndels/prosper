#ifndef PROSPER_LIGHT_CLUSTERING_HPP
#define PROSPER_LIGHT_CLUSTERING_HPP

#include <functional>

#include "Camera.hpp"
#include "Device.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "World.hpp"

class LightClustering
{
  public:
    LightClustering(
        Device *device, RenderResources *resources,
        const SwapchainConfig &swapConfig,
        const vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    ~LightClustering();

    LightClustering(const LightClustering &other) = delete;
    LightClustering &operator=(const LightClustering &other) = delete;

    void recompileShaders(
        const vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    void recreateSwapchainRelated(
        const SwapchainConfig &swapConfig,
        const vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    vk::CommandBuffer recordCommandBuffer(
        const Scene &scene, const Camera &cam, const vk::Rect2D &renderArea,
        const uint32_t nextImage);

  private:
    bool compileShaders();

    void destroySwapchainRelated();
    void destroyPipeline();
    // These also need to be recreated with Swapchain as they depend on
    // swapconfig
    void createOutputs(const SwapchainConfig &swapConfig);
    void createDescriptorSets(const SwapchainConfig &swapConfig);
    void createPipeline(
        const vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    void createCommandBuffers(const SwapchainConfig &swapConfig);

    Device *_device = nullptr;
    RenderResources *_resources = nullptr;

    vk::ShaderModule _compSM;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;

    std::vector<vk::CommandBuffer> _commandBuffers;
};

#endif // PROSPER_LIGHT_CLUSTERING_HPP