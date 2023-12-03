#include "RtDirectIllumination.hpp"

#include "../../utils/Profiler.hpp"
#include "../RenderResources.hpp"

#include <imgui.h>

using namespace wheels;

RtDirectIllumination::RtDirectIllumination(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    vk::DescriptorSetLayout camDSLayout, const WorldDSLayouts &worldDSLayouts)
: _resources{resources}
, _initialReservoirs{
      scopeAlloc.child_scope(), device, resources, staticDescriptorsAlloc,
      RtDiInitialReservoirs::InputDSLayouts{
          .camera = camDSLayout,
          .world = worldDSLayouts,
      }}
, _spatialReuse{
      scopeAlloc.child_scope(), device, resources, staticDescriptorsAlloc,
      RtDiSpatialReuse::InputDSLayouts{
          .camera = camDSLayout,
          .world = worldDSLayouts,
      }}
, _trace{
      scopeAlloc.child_scope(), device, resources, staticDescriptorsAlloc,
           camDSLayout,
           worldDSLayouts
     }
{
    WHEELS_ASSERT(_resources != nullptr);
}

void RtDirectIllumination::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    vk::DescriptorSetLayout camDSLayout, const WorldDSLayouts &worldDSLayouts)
{
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
    ImGui::Checkbox("Spatial reuse", &_doSpatialReuse);
    _trace.drawUi();
}

RtDirectIllumination::Output RtDirectIllumination::record(
    ScopedScratch scopeAlloc,vk::CommandBuffer cb, const World &world, const Camera &cam,
    const GBufferRendererOutput &gbuffer, bool resetAccumulation,
    uint32_t nextFrame, Profiler *profiler)
{
    Output ret;
    {
        const auto _s = profiler->createCpuGpuScope(cb, "RtDirectIllumination");

        const RtDiInitialReservoirs::Output initialReservoirsOutput =
            _initialReservoirs.record(
              scopeAlloc.child_scope(),   cb, world, cam, gbuffer, nextFrame, profiler);

        ImageHandle reservoirs = initialReservoirsOutput.reservoirs;
        if (_doSpatialReuse)
        {
            const RtDiSpatialReuse::Output spatialReuseOutput =
                _spatialReuse.record(
 scopeAlloc.child_scope(),
                    cb, world, cam,
                    RtDiSpatialReuse::Input{
                        .gbuffer = gbuffer,
                        .reservoirs = initialReservoirsOutput.reservoirs,
                    },
                    nextFrame, profiler);

            _resources->images.release(initialReservoirsOutput.reservoirs);
            reservoirs = spatialReuseOutput.reservoirs;
        }

        ret = _trace.record(
            scopeAlloc.child_scope(), cb, world, cam,
            RtDiTrace::Input{
                .gbuffer = gbuffer,
                .reservoirs = reservoirs,
            },
            resetAccumulation || _resetAccumulation, nextFrame, profiler);

        _resources->images.release(reservoirs);
    }

    _resetAccumulation = false;

    return ret;
}

void RtDirectIllumination::releasePreserved() { _trace.releasePreserved(); }
