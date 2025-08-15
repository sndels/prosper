#include "RtDirectIllumination.hpp"

#include "render/RenderResources.hpp"
#include "utils/Profiler.hpp"

#include <imgui.h>

using namespace wheels;

namespace render::rtdi
{

void RtDirectIllumination::init(
    ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDSLayout,
    const scene::WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(!m_initialized);

    m_initialReservoirs.init(
        scopeAlloc.child_scope(), RtDiInitialReservoirs::InputDSLayouts{
                                      .camera = camDSLayout,
                                      .world = worldDSLayouts,
                                  });
    m_spatialReuse.init(
        scopeAlloc.child_scope(), RtDiSpatialReuse::InputDSLayouts{
                                      .camera = camDSLayout,
                                      .world = worldDSLayouts,
                                  });
    m_trace.init(WHEELS_MOV(scopeAlloc), camDSLayout, worldDSLayouts);

    m_initialized = true;
}

void RtDirectIllumination::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    vk::DescriptorSetLayout camDSLayout,
    const scene::WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(m_initialized);

    m_resetAccumulation |= m_initialReservoirs.recompileShaders(
        scopeAlloc.child_scope(), changedFiles,
        RtDiInitialReservoirs::InputDSLayouts{
            .camera = camDSLayout,
            .world = worldDSLayouts,
        });
    m_resetAccumulation |= m_spatialReuse.recompileShaders(
        scopeAlloc.child_scope(), changedFiles,
        RtDiSpatialReuse::InputDSLayouts{
            .camera = camDSLayout,
            .world = worldDSLayouts,
        });
    // Trace handles accumulation so we don't check recompile here
    m_trace.recompileShaders(
        scopeAlloc.child_scope(), changedFiles, camDSLayout, worldDSLayouts);
}

void RtDirectIllumination::drawUi()
{
    WHEELS_ASSERT(m_initialized);

    ImGui::Checkbox("Spatial reuse", &m_doSpatialReuse);
}

RtDirectIllumination::Output RtDirectIllumination::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, scene::World &world,
    const scene::Camera &cam, const GBufferRendererOutput &gbuffer,
    bool resetAccumulation, scene::DrawType drawType, uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_GPU_SCOPE(cb, "RtDirectIllumination");

    Output ret;
    {

        const RtDiInitialReservoirs::Output initialReservoirsOutput =
            m_initialReservoirs.record(
                scopeAlloc.child_scope(), cb, world, cam, gbuffer, nextFrame);

        ImageHandle reservoirs = initialReservoirsOutput.reservoirs;
        if (m_doSpatialReuse)
        {
            const RtDiSpatialReuse::Output spatialReuseOutput =
                m_spatialReuse.record(
                    scopeAlloc.child_scope(), cb, world, cam,
                    RtDiSpatialReuse::Input{
                        .gbuffer = gbuffer,
                        .reservoirs = initialReservoirsOutput.reservoirs,
                    },
                    nextFrame);

            gRenderResources.images->release(
                initialReservoirsOutput.reservoirs);
            reservoirs = spatialReuseOutput.reservoirs;
        }

        ret = m_trace.record(
            scopeAlloc.child_scope(), cb, world, cam,
            RtDiTrace::Input{
                .gbuffer = gbuffer,
                .reservoirs = reservoirs,
            },
            resetAccumulation || m_resetAccumulation, drawType, nextFrame);

        gRenderResources.images->release(reservoirs);
    }

    m_resetAccumulation = false;

    return ret;
}

void RtDirectIllumination::releasePreserved()
{
    WHEELS_ASSERT(m_initialized);

    m_trace.releasePreserved();
}

} // namespace render::rtdi
