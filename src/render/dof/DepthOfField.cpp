#include "DepthOfField.hpp"

#include "../../utils/Profiler.hpp"
#include "../RenderResources.hpp"

using namespace wheels;

void DepthOfField::init(
    ScopedScratch scopeAlloc, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    vk::DescriptorSetLayout cameraDsLayout)
{
    WHEELS_ASSERT(!_initialized);
    WHEELS_ASSERT(resources != nullptr);

    _resources = resources;
    _setupPass.init(
        scopeAlloc.child_scope(), resources, staticDescriptorsAlloc,
        cameraDsLayout);
    _reducePass.init(
        scopeAlloc.child_scope(), resources, staticDescriptorsAlloc);
    _flattenPass.init(
        scopeAlloc.child_scope(), resources, staticDescriptorsAlloc);
    _dilatePass.init(
        scopeAlloc.child_scope(), resources, staticDescriptorsAlloc);
    _gatherPass.init(
        scopeAlloc.child_scope(), resources, staticDescriptorsAlloc);
    _filterPass.init(
        scopeAlloc.child_scope(), resources, staticDescriptorsAlloc);
    _combinePass.init(
        scopeAlloc.child_scope(), resources, staticDescriptorsAlloc);

    _initialized = true;
}

void DepthOfField::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    vk::DescriptorSetLayout cameraDsLayout)
{
    WHEELS_ASSERT(_initialized);

    _setupPass.recompileShaders(
        scopeAlloc.child_scope(), changedFiles, cameraDsLayout);
    _reducePass.recompileShaders(scopeAlloc.child_scope(), changedFiles);
    _flattenPass.recompileShaders(scopeAlloc.child_scope(), changedFiles);
    _dilatePass.recompileShaders(scopeAlloc.child_scope(), changedFiles);
    _gatherPass.recompileShaders(scopeAlloc.child_scope(), changedFiles);
    _filterPass.recompileShaders(scopeAlloc.child_scope(), changedFiles);
    _combinePass.recompileShaders(scopeAlloc.child_scope(), changedFiles);
}

void DepthOfField::startFrame() { _filterPass.startFrame(); }

DepthOfField::Output DepthOfField::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Camera &cam,
    const DepthOfField::Input &input, uint32_t nextFrame, Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);

    Output ret;
    {
        const auto _s = profiler->createCpuGpuScope(cb, "DepthOfField");

        const DepthOfFieldSetup::Output setupOutput = _setupPass.record(
            scopeAlloc.child_scope(), cb, cam, input, nextFrame, profiler);

        _reducePass.record(
            scopeAlloc.child_scope(), cb, setupOutput.halfResIllumination,
            nextFrame, profiler);

        const DepthOfFieldFlatten::Output flattenOutput = _flattenPass.record(
            scopeAlloc.child_scope(), cb, setupOutput.halfResCircleOfConfusion,
            nextFrame, profiler);

        const DepthOfFieldDilate::Output dilateOutput = _dilatePass.record(
            scopeAlloc.child_scope(), cb,
            flattenOutput.tileMinMaxCircleOfConfusion, cam, nextFrame,
            profiler);

        _resources->images.release(flattenOutput.tileMinMaxCircleOfConfusion);

        const DepthOfFieldGather::Input gatherInput{
            .halfResIllumination = setupOutput.halfResIllumination,
            .halfResCoC = setupOutput.halfResCircleOfConfusion,
            .dilatedTileMinMaxCoC = dilateOutput.dilatedTileMinMaxCoC,
        };
        const DepthOfFieldGather::Output fgGatherOutput = _gatherPass.record(
            scopeAlloc.child_scope(), cb, gatherInput,
            DepthOfFieldGather::GatherType_Foreground, nextFrame, profiler);
        const DepthOfFieldGather::Output bgGatherOutput = _gatherPass.record(
            scopeAlloc.child_scope(), cb, gatherInput,
            DepthOfFieldGather::GatherType_Background, nextFrame, profiler);

        const DepthOfFieldFilter::Output fgFilterOutput = _filterPass.record(
            scopeAlloc.child_scope(), cb,
            fgGatherOutput.halfResBokehColorWeight, nextFrame,
            DepthOfFieldFilter::DebugNames{
                .scope = "  FilterFG",
                .outRes = "halfResFgColorWeightdFiltered",
            },
            profiler);
        _resources->images.release(fgGatherOutput.halfResBokehColorWeight);
        const DepthOfFieldFilter::Output bgFilterOutput = _filterPass.record(
            scopeAlloc.child_scope(), cb,
            bgGatherOutput.halfResBokehColorWeight, nextFrame,
            DepthOfFieldFilter::DebugNames{
                .scope = "  FilterBG",
                .outRes = "halfResBgColorWeightdFiltered",
            },
            profiler);
        _resources->images.release(bgGatherOutput.halfResBokehColorWeight);

        ret = _combinePass.record(
            scopeAlloc.child_scope(), cb,
            DepthOfFieldCombine::Input{
                .halfResFgBokehWeight =
                    fgFilterOutput.filteredIlluminationWeight,
                .halfResBgBokehWeight =
                    bgFilterOutput.filteredIlluminationWeight,
                .halfResCircleOfConfusion =
                    setupOutput.halfResCircleOfConfusion,
                .illumination = input.illumination,
            },
            nextFrame, profiler);

        _resources->images.release(bgFilterOutput.filteredIlluminationWeight);
        _resources->images.release(fgFilterOutput.filteredIlluminationWeight);
        _resources->images.release(dilateOutput.dilatedTileMinMaxCoC);
        _resources->images.release(setupOutput.halfResIllumination);
        _resources->images.release(setupOutput.halfResCircleOfConfusion);
    }
    return ret;
}
