#include "DepthOfField.hpp"

using namespace wheels;

DepthOfField::DepthOfField(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    vk::DescriptorSetLayout cameraDsLayout)
: _resources{resources}
, _setupPass{scopeAlloc.child_scope(), device, resources, staticDescriptorsAlloc, cameraDsLayout}
, _flattenPass{scopeAlloc.child_scope(), device, resources, staticDescriptorsAlloc}
, _dilatePass{scopeAlloc.child_scope(), device, resources, staticDescriptorsAlloc}
, _gatherPass{scopeAlloc.child_scope(), device, resources, staticDescriptorsAlloc}
, _combinePass{
      scopeAlloc.child_scope(), device, resources, staticDescriptorsAlloc}
{
    assert(_resources != nullptr);
}

void DepthOfField::recompileShaders(
    wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout cameraDsLayout)
{
    _setupPass.recompileShaders(scopeAlloc.child_scope(), cameraDsLayout);
    _flattenPass.recompileShaders(scopeAlloc.child_scope());
    _dilatePass.recompileShaders(scopeAlloc.child_scope());
    _gatherPass.recompileShaders(scopeAlloc.child_scope());
    _combinePass.recompileShaders(scopeAlloc.child_scope());
}

DepthOfField::Output DepthOfField::record(
    vk::CommandBuffer cb, const Camera &cam, const DepthOfField::Input &input,
    uint32_t nextFrame, Profiler *profiler)
{
    Output ret;
    {
        const auto _s = profiler->createCpuGpuScope(cb, "DepthOfField");

        const DepthOfFieldSetup::Output setupOutput =
            _setupPass.record(cb, cam, input, nextFrame, profiler);

        const DepthOfFieldFlatten::Output flattenOutput = _flattenPass.record(
            cb, setupOutput.halfResCircleOfConfusion, nextFrame, profiler);

        const DepthOfFieldDilate::Output dilateOutput = _dilatePass.record(
            cb, flattenOutput.tileMinMaxCircleOfConfusion, nextFrame, profiler);

        _resources->images.release(flattenOutput.tileMinMaxCircleOfConfusion);

        const DepthOfFieldGather::Input gatherInput{
            .halfResIllumination = setupOutput.halfResIllumination,
            .halfResCoC = setupOutput.halfResCircleOfConfusion,
            .dilatedTileMinMaxCoC = dilateOutput.dilatedTileMinMaxCoC,
        };
        const DepthOfFieldGather::Output fgGatherOutput = _gatherPass.record(
            cb, gatherInput, DepthOfFieldGather::GatherType_Foreground,
            nextFrame, profiler);
        const DepthOfFieldGather::Output bgGatherOutput = _gatherPass.record(
            cb, gatherInput, DepthOfFieldGather::GatherType_Background,
            nextFrame, profiler);

        ret = _combinePass.record(
            cb,
            DepthOfFieldCombine::Input{
                .halfResFgBokehWeight = fgGatherOutput.halfResBokehColorWeight,
                .halfResBgBokehWeight = bgGatherOutput.halfResBokehColorWeight,
                .halfResCircleOfConfusion =
                    setupOutput.halfResCircleOfConfusion,
                .illumination = input.illumination,
            },
            nextFrame, profiler);

        _resources->images.release(bgGatherOutput.halfResBokehColorWeight);
        _resources->images.release(fgGatherOutput.halfResBokehColorWeight);
        _resources->images.release(dilateOutput.dilatedTileMinMaxCoC);
        _resources->images.release(setupOutput.halfResIllumination);
        _resources->images.release(setupOutput.halfResCircleOfConfusion);
    }
    return ret;
}
