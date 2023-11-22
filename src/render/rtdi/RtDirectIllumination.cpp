#include "RtDirectIllumination.hpp"

using namespace wheels;

RtDirectIllumination::RtDirectIllumination(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    vk::DescriptorSetLayout camDSLayout, const World::DSLayouts &worldDSLayouts)
: _resources{resources}
, _initialReservoirs{
      scopeAlloc.child_scope(), device, resources, staticDescriptorsAlloc,
      RtDiInitialReservoirs::InputDSLayouts{
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
    vk::DescriptorSetLayout camDSLayout, const World::DSLayouts &worldDSLayouts)
{
    _initialReservoirs.recompileShaders(
        scopeAlloc.child_scope(), changedFiles,
        RtDiInitialReservoirs::InputDSLayouts{
            .camera = camDSLayout,
            .world = worldDSLayouts,
        });
    _trace.recompileShaders(
        scopeAlloc.child_scope(), changedFiles, camDSLayout, worldDSLayouts);
}

void RtDirectIllumination::drawUi() { _trace.drawUi(); }

RtDirectIllumination::Output RtDirectIllumination::record(
    vk::CommandBuffer cb, const World &world, const Camera &cam,
    const GBufferRenderer::Output &gbuffer, bool resetAccumulation,
    uint32_t nextFrame, Profiler *profiler)
{
    Output ret;
    {
        const auto _s = profiler->createCpuGpuScope(cb, "RtDirectIllumination");

        const RtDiInitialReservoirs::Output initialReservoirsOutput =
            _initialReservoirs.record(
                cb, world, cam, gbuffer, nextFrame, profiler);

        ret = _trace.record(
            cb, world, cam,
            RtDiTrace::Input{
                .gbuffer = gbuffer,
                .reservoirs = initialReservoirsOutput.reservoirs,
            },
            resetAccumulation, nextFrame, profiler);

        _resources->images.release(initialReservoirsOutput.reservoirs);
    }
    return ret;
}

void RtDirectIllumination::releasePreserved() { _trace.releasePreserved(); }
