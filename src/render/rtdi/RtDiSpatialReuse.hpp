#ifndef PROSPER_RENDER_RTDI_SPATIAL_REUSE_HPP
#define PROSPER_RENDER_RTDI_SPATIAL_REUSE_HPP

#include "gfx/Fwd.hpp"
#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "scene/Fwd.hpp"
#include "utils/Fwd.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

class RtDiSpatialReuse
{
  public:
    RtDiSpatialReuse() noexcept = default;
    ~RtDiSpatialReuse() = default;

    RtDiSpatialReuse(const RtDiSpatialReuse &other) = delete;
    RtDiSpatialReuse(RtDiSpatialReuse &&other) = delete;
    RtDiSpatialReuse &operator=(const RtDiSpatialReuse &other) = delete;
    RtDiSpatialReuse &operator=(RtDiSpatialReuse &&other) = delete;

    struct InputDSLayouts
    {
        vk::DescriptorSetLayout camera;
        const WorldDSLayouts &world;
    };
    void init(
        wheels::ScopedScratch scopeAlloc, const InputDSLayouts &dsLayouts);

    // Returns true if recompile happened
    bool recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        const InputDSLayouts &dsLayouts);

    struct Input
    {
        const GBufferRendererOutput &gbuffer;
        ImageHandle reservoirs;
    };
    struct Output
    {
        ImageHandle reservoirs;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const World &world, const Camera &cam, const Input &input,
        uint32_t nextFrame);

    bool m_initialized{false};
    ComputePass m_computePass;

    uint32_t m_frameIndex{0};
};

#endif // PROSPER_RENDER_RT_DI_SPATIAL_REUSE_HPP
