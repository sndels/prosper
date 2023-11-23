#ifndef PROSPER_RENDER_RTDI_RT_DIRECT_ILLUMINATION_HPP
#define PROSPER_RENDER_RTDI_RT_DIRECT_ILLUMINATION_HPP

#include "../../gfx/Device.hpp"
#include "../../scene/Camera.hpp"
#include "../../scene/World.hpp"
#include "../../utils/Profiler.hpp"
#include "../GBufferRenderer.hpp"
#include "../RenderResources.hpp"
#include "RtDiInitialReservoirs.hpp"
#include "RtDiSpatialReuse.hpp"
#include "RtDiTrace.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

class RtDirectIllumination
{
  public:
    enum class DrawType : uint32_t
    {
        Default = 0,
        DEBUG_DRAW_TYPES_AND_COUNT
    };

    RtDirectIllumination(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc,
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    ~RtDirectIllumination() = default;

    RtDirectIllumination(const RtDirectIllumination &other) = delete;
    RtDirectIllumination(RtDirectIllumination &&other) = delete;
    RtDirectIllumination &operator=(const RtDirectIllumination &other) = delete;
    RtDirectIllumination &operator=(RtDirectIllumination &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    void drawUi();

    using Output = RtDiTrace::Output;
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, const World &world, const Camera &cam,
        const GBufferRenderer::Output &gbuffer, bool resetAccumulation,
        uint32_t nextFrame, Profiler *profiler);
    void releasePreserved();

  private:
    RenderResources *_resources{nullptr};

    bool _doSpatialReuse{true};

    RtDiInitialReservoirs _initialReservoirs;
    RtDiSpatialReuse _spatialReuse;
    RtDiTrace _trace;
};

#endif // PROSPER_RENDER_RTDI_RT_DIRECT_ILLUMINATION_HPP
