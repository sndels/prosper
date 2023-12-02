#ifndef PROSPER_RENDER_LIGHT_CLUSTERING_HPP
#define PROSPER_RENDER_LIGHT_CLUSTERING_HPP

#include "../gfx/Device.hpp"
#include "../scene/Camera.hpp"
#include "../scene/World.hpp"
#include "../utils/Profiler.hpp"
#include "ComputePass.hpp"
#include "RenderResources.hpp"

#include <wheels/allocators/scoped_scratch.hpp>

struct LightClusteringOutput
{
    ImageHandle pointers;
    TexelBufferHandle indicesCount;
    TexelBufferHandle indices;
    vk::DescriptorSet descriptorSet;
};

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
        RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc,
        vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);

    ~LightClustering() = default;

    LightClustering(const LightClustering &other) = delete;
    LightClustering(LightClustering &&other) = delete;
    LightClustering &operator=(const LightClustering &other) = delete;
    LightClustering &operator=(LightClustering &&other) = delete;

    [[nodiscard]] vk::DescriptorSetLayout descriptorSetLayout() const;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);

    [[nodiscard]] LightClusteringOutput record(
        vk::CommandBuffer cb, const World &world, const Camera &cam,
        const vk::Extent2D &renderExtent, uint32_t nextFrame,
        Profiler *profiler);

  private:
    [[nodiscard]] LightClusteringOutput createOutputs(
        const vk::Extent2D &renderExtent);

    RenderResources *_resources{nullptr};
    ComputePass _computePass;
};

#endif // PROSPER_RENDER_LIGHT_CLUSTERING_HPP
