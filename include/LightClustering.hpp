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
        RenderResources *resources, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    ~LightClustering();

    LightClustering(const LightClustering &other) = delete;
    LightClustering(LightClustering &&other) = delete;
    LightClustering &operator=(const LightClustering &other) = delete;
    LightClustering &operator=(LightClustering &&other) = delete;

    [[nodiscard]] vk::DescriptorSetLayout descriptorSetLayout() const;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    struct Output
    {
        ImageHandle pointers;
        TexelBufferHandle indicesCount;
        TexelBufferHandle indices;
        vk::DescriptorSet descriptorSet;
    } lightClusters;
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, const Scene &scene, const Camera &cam,
        const vk::Extent2D &renderExtent, uint32_t nextFrame,
        Profiler *profiler);

  private:
    [[nodiscard]] bool compileShaders(wheels::ScopedScratch scopeAlloc);

    void recordBarriers(vk::CommandBuffer cb, const Output &output) const;

    void destroyPipeline();

    [[nodiscard]] Output createOutputs(const vk::Extent2D &renderExtent);
    void createDescriptorSets();
    void updateDescriptorSet(uint32_t nextFrame, Output &outputs);
    void createPipeline(
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    vk::ShaderModule _compSM;

    vk::DescriptorSetLayout _descriptorSetLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
        _descriptorSets{{}};

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
};

#endif // PROSPER_LIGHT_CLUSTERING_HPP
