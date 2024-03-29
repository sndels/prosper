#ifndef PROSPER_RENDER_RTDI_RT_DIRECT_ILLUMINATION_HPP
#define PROSPER_RENDER_RTDI_RT_DIRECT_ILLUMINATION_HPP

#include "../../gfx/Fwd.hpp"
#include "../../scene/DrawType.hpp"
#include "../../scene/Fwd.hpp"
#include "../../utils/Fwd.hpp"
#include "../Fwd.hpp"
#include "../RenderResourceHandle.hpp"
#include "RtDiInitialReservoirs.hpp"
#include "RtDiSpatialReuse.hpp"
#include "RtDiTrace.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

class RtDirectIllumination
{
  public:
    RtDirectIllumination() noexcept = default;
    ~RtDirectIllumination() = default;

    RtDirectIllumination(const RtDirectIllumination &other) = delete;
    RtDirectIllumination(RtDirectIllumination &&other) = delete;
    RtDirectIllumination &operator=(const RtDirectIllumination &other) = delete;
    RtDirectIllumination &operator=(RtDirectIllumination &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc,
        vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);

    void drawUi();

    using Output = RtDiTrace::Output;
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb, World &world,
        const Camera &cam, const GBufferRendererOutput &gbuffer,
        bool resetAccumulation, DrawType drawType, uint32_t nextFrame,
        Profiler *profiler);
    void releasePreserved();

  private:
    bool _initialized{false};
    RenderResources *_resources{nullptr};

    bool _doSpatialReuse{true};
    bool _resetAccumulation{true};

    RtDiInitialReservoirs _initialReservoirs;
    RtDiSpatialReuse _spatialReuse;
    RtDiTrace _trace;
};

#endif // PROSPER_RENDER_RTDI_RT_DIRECT_ILLUMINATION_HPP
