#ifndef PROSPER_RENDER_RTDI_RT_DIRECT_ILLUMINATION_HPP
#define PROSPER_RENDER_RTDI_RT_DIRECT_ILLUMINATION_HPP

#include "Compose.hpp"
#include "Trace.hpp"
#include "render/Fwd.hpp"
#include "render/rtdi/InitialReservoirs.hpp"
#include "render/rtdi/SpatialReuse.hpp"
#include "scene/DrawType.hpp"
#include "scene/Fwd.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

namespace render::rtdi
{

using Output = Compose::Output;

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
        wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDSLayout,
        const scene::WorldDSLayouts &worldDSLayouts);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDSLayout,
        const scene::WorldDSLayouts &worldDSLayouts);

    void drawUi();

    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        scene::World &world, const scene::Camera &cam, const GBuffer &gbuffer,
        bool resetAccumulation, scene::DrawType drawType, uint32_t nextFrame);
    void releasePreserved();

  private:
    bool m_initialized{false};

    bool m_doSpatialReuse{true};
    bool m_resetAccumulation{true};

    InitialReservoirs m_initialReservoirs;
    SpatialReuse m_spatialReuse;
    Trace m_trace;
    Compose m_compose;
};

} // namespace render::rtdi

#endif // PROSPER_RENDER_RTDI_RT_DIRECT_ILLUMINATION_HPP
