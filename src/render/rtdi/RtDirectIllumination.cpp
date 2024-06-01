#include "RtDirectIllumination.hpp"

#include "../../utils/Profiler.hpp"
#include "../RenderResources.hpp"

#include <imgui.h>

using namespace wheels;

void RtDirectIllumination::init(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc,
    vk::DescriptorSetLayout camDSLayout, const WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(!_initialized);

    _initialReservoirs.init(
        scopeAlloc.child_scope(), staticDescriptorsAlloc,
        RtDiInitialReservoirs::InputDSLayouts{
            .camera = camDSLayout,
            .world = worldDSLayouts,
        });
    _spatialReuse.init(
        scopeAlloc.child_scope(), staticDescriptorsAlloc,
        RtDiSpatialReuse::InputDSLayouts{
            .camera = camDSLayout,
            .world = worldDSLayouts,
        });
    _trace.init(
        WHEELS_MOV(scopeAlloc), staticDescriptorsAlloc, camDSLayout,
        worldDSLayouts);

    _initialized = true;
}

void RtDirectIllumination::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    vk::DescriptorSetLayout camDSLayout, const WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(_initialized);

    _resetAccumulation |= _initialReservoirs.recompileShaders(
        scopeAlloc.child_scope(), changedFiles,
        RtDiInitialReservoirs::InputDSLayouts{
            .camera = camDSLayout,
            .world = worldDSLayouts,
        });
    _resetAccumulation |= _spatialReuse.recompileShaders(
        scopeAlloc.child_scope(), changedFiles,
        RtDiSpatialReuse::InputDSLayouts{
            .camera = camDSLayout,
            .world = worldDSLayouts,
        });
    // Trace handles accumulation so we don't check recompile here
    _trace.recompileShaders(
        scopeAlloc.child_scope(), changedFiles, camDSLayout, worldDSLayouts);
}

void RtDirectIllumination::drawUi()
{
    WHEELS_ASSERT(_initialized);

    ImGui::Checkbox("Spatial reuse", &_doSpatialReuse);
}

RtDirectIllumination::Output RtDirectIllumination::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, World &world,
    const Camera &cam, const GBufferRendererOutput &gbuffer,
    bool resetAccumulation, DrawType drawType, uint32_t nextFrame,
    Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);

    Output ret;
    {
        const auto _s = profiler->createCpuGpuScope(cb, "RtDirectIllumination");

        const RtDiInitialReservoirs::Output initialReservoirsOutput =
            _initialReservoirs.record(
                scopeAlloc.child_scope(), cb, world, cam, gbuffer, nextFrame,
                profiler);

        ImageHandle reservoirs = initialReservoirsOutput.reservoirs;
        if (_doSpatialReuse)
        {
            const RtDiSpatialReuse::Output spatialReuseOutput =
                _spatialReuse.record(
                    scopeAlloc.child_scope(), cb, world, cam,
                    RtDiSpatialReuse::Input{
                        .gbuffer = gbuffer,
                        .reservoirs = initialReservoirsOutput.reservoirs,
                    },
                    nextFrame, profiler);

            gRenderResources.images->release(
                initialReservoirsOutput.reservoirs);
            reservoirs = spatialReuseOutput.reservoirs;
        }

        ret = _trace.record(
            scopeAlloc.child_scope(), cb, world, cam,
            RtDiTrace::Input{
                .gbuffer = gbuffer,
                .reservoirs = reservoirs,
            },
            resetAccumulation || _resetAccumulation, drawType, nextFrame,
            profiler);

        gRenderResources.images->release(reservoirs);
    }

    _resetAccumulation = false;

    return ret;
}

void RtDirectIllumination::releasePreserved()
{
    WHEELS_ASSERT(_initialized);

    _trace.releasePreserved();
}
