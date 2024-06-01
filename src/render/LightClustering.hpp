#ifndef PROSPER_RENDER_LIGHT_CLUSTERING_HPP
#define PROSPER_RENDER_LIGHT_CLUSTERING_HPP

#include "../gfx/Fwd.hpp"
#include "../scene/Fwd.hpp"
#include "../utils/Fwd.hpp"
#include "ComputePass.hpp"
#include "Fwd.hpp"
#include "RenderResourceHandle.hpp"

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

    LightClustering() noexcept = default;
    ~LightClustering() = default;

    LightClustering(const LightClustering &other) = delete;
    LightClustering(LightClustering &&other) = delete;
    LightClustering &operator=(const LightClustering &other) = delete;
    LightClustering &operator=(LightClustering &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc, RenderResources *resources,
        DescriptorAllocator *staticDescriptorsAlloc,
        vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);

    [[nodiscard]] vk::DescriptorSetLayout descriptorSetLayout() const;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);

    [[nodiscard]] LightClusteringOutput record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const World &world, const Camera &cam, const vk::Extent2D &renderExtent,
        uint32_t nextFrame, Profiler *profiler);

  private:
    [[nodiscard]] LightClusteringOutput createOutputs(
        const vk::Extent2D &renderExtent);

    bool _initialized{false};
    RenderResources *_resources{nullptr};
    ComputePass _computePass;
};

#endif // PROSPER_RENDER_LIGHT_CLUSTERING_HPP
