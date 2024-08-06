#ifndef PROSPER_RENDER_DEFERRED_SHADING_HPP
#define PROSPER_RENDER_DEFERRED_SHADING_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "scene/DrawType.hpp"
#include "scene/Fwd.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

class DeferredShading
{
  public:
    DeferredShading() noexcept = default;
    ~DeferredShading() = default;

    DeferredShading(const DeferredShading &other) = delete;
    DeferredShading(DeferredShading &&other) = delete;
    DeferredShading &operator=(const DeferredShading &other) = delete;
    DeferredShading &operator=(DeferredShading &&other) = delete;

    struct InputDSLayouts
    {
        vk::DescriptorSetLayout camera;
        vk::DescriptorSetLayout lightClusters;
        const WorldDSLayouts &world;
    };
    void init(
        wheels::ScopedScratch scopeAlloc, const InputDSLayouts &dsLayouts);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        const InputDSLayouts &dsLayouts);

    struct Input
    {
        const GBufferRendererOutput &gbuffer;
        const LightClusteringOutput &lightClusters;
    };
    struct Output
    {
        ImageHandle illumination;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const World &world, const Camera &cam, const Input &input,
        uint32_t nextFrame, bool applyIbl, DrawType drawType);

    bool m_initialized{false};
    ComputePass m_computePass;
};

#endif // PROSPER_RENDER_DEFERRED_SHADING_HPP
