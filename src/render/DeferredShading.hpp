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

namespace render
{

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
        const scene::WorldDSLayouts &world;
    };
    void init(
        wheels::ScopedScratch scopeAlloc, const InputDSLayouts &dsLayouts);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        const InputDSLayouts &dsLayouts);

    struct Input
    {
        const GBuffer &gbuffer;
        const LightClusteringOutput &lightClusters;
    };
    struct Output
    {
        ImageHandle illumination;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const scene::World &world, const scene::Camera &cam, const Input &input,
        uint32_t nextFrame, bool applyIbl, scene::DrawType drawType);

    bool m_initialized{false};
    ComputePass m_computePass;
};

} // namespace render

#endif // PROSPER_RENDER_DEFERRED_SHADING_HPP
