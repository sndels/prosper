#ifndef PROSPER_LIGHT_CLUSTERING_HPP
#define PROSPER_LIGHT_CLUSTERING_HPP

#include "Camera.hpp"
#include "Device.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "World.hpp"

#include <wheels/allocators/scoped_scratch.hpp>

class LightClustering
{
  public:
    static const uint32_t clusterDim = 32;
    static const uint32_t zSlices = 16;
    static void appendShaderDefines(wheels::String &str)
    {
        appendDefineStr(str, "LIGHT_CLUSTER_DIMENSION", clusterDim);
        appendDefineStr(str, "LIGHT_CLUSTER_Z_SLICE_COUNT", zSlices);
    };

    LightClustering(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, const SwapchainConfig &swapConfig,
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    ~LightClustering();

    LightClustering(const LightClustering &other) = delete;
    LightClustering(LightClustering &&other) = delete;
    LightClustering &operator=(const LightClustering &other) = delete;
    LightClustering &operator=(LightClustering &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    void recreate(
        const SwapchainConfig &swapConfig, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    void record(
        vk::CommandBuffer cb, const Scene &scene, const Camera &cam,
        const vk::Rect2D &renderArea, uint32_t nextImage, Profiler *profiler);

  private:
    [[nodiscard]] bool compileShaders(wheels::ScopedScratch scopeAlloc);

    void destroySwapchainRelated();
    void destroyPipeline();
    // These also need to be recreated with Swapchain as they depend on
    // swapconfig
    void createOutputs(const SwapchainConfig &swapConfig);
    void createDescriptorSets(const SwapchainConfig &swapConfig);
    void createPipeline(
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    vk::ShaderModule _compSM;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
};

#endif // PROSPER_LIGHT_CLUSTERING_HPP
