#ifndef PROSPER_RENDER_LIGHT_CLUSTERING_HPP
#define PROSPER_RENDER_LIGHT_CLUSTERING_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "scene/Fwd.hpp"

#include <wheels/allocators/scoped_scratch.hpp>

namespace render
{

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
        wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDSLayout,
        const scene::WorldDSLayouts &worldDSLayouts);

    [[nodiscard]] vk::DescriptorSetLayout descriptorSetLayout() const;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDSLayout,
        const scene::WorldDSLayouts &worldDSLayouts);

    [[nodiscard]] LightClusteringOutput record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const scene::World &world, const scene::Camera &cam,
        const vk::Extent2D &renderExtent, uint32_t nextFrame);

  private:
    bool m_initialized{false};
    ComputePass m_computePass;
};

} // namespace render

#endif // PROSPER_RENDER_LIGHT_CLUSTERING_HPP
