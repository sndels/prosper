#include "DepthOfField.hpp"

#include "../../utils/Profiler.hpp"
#include "../RenderResources.hpp"

using namespace wheels;

void DepthOfField::init(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc,
    vk::DescriptorSetLayout cameraDsLayout)
{
    WHEELS_ASSERT(!m_initialized);

    m_setupPass.init(
        scopeAlloc.child_scope(), staticDescriptorsAlloc, cameraDsLayout);
    m_reducePass.init(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    m_flattenPass.init(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    m_dilatePass.init(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    m_gatherPass.init(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    m_filterPass.init(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    m_combinePass.init(scopeAlloc.child_scope(), staticDescriptorsAlloc);

    m_initialized = true;
}

void DepthOfField::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    vk::DescriptorSetLayout cameraDsLayout)
{
    WHEELS_ASSERT(m_initialized);

    m_setupPass.recompileShaders(
        scopeAlloc.child_scope(), changedFiles, cameraDsLayout);
    m_reducePass.recompileShaders(scopeAlloc.child_scope(), changedFiles);
    m_flattenPass.recompileShaders(scopeAlloc.child_scope(), changedFiles);
    m_dilatePass.recompileShaders(scopeAlloc.child_scope(), changedFiles);
    m_gatherPass.recompileShaders(scopeAlloc.child_scope(), changedFiles);
    m_filterPass.recompileShaders(scopeAlloc.child_scope(), changedFiles);
    m_combinePass.recompileShaders(scopeAlloc.child_scope(), changedFiles);
}

void DepthOfField::startFrame() { m_filterPass.startFrame(); }

DepthOfField::Output DepthOfField::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Camera &cam,
    const DepthOfField::Input &input, uint32_t nextFrame, Profiler *profiler)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_GPU_SCOPE(profiler, cb, "DepthOfField");

    Output ret;
    {

        const DepthOfFieldSetup::Output setupOutput = m_setupPass.record(
            scopeAlloc.child_scope(), cb, cam, input, nextFrame, profiler);

        m_reducePass.record(
            scopeAlloc.child_scope(), cb, setupOutput.halfResIllumination,
            nextFrame, profiler);

        const DepthOfFieldFlatten::Output flattenOutput = m_flattenPass.record(
            scopeAlloc.child_scope(), cb, setupOutput.halfResCircleOfConfusion,
            nextFrame, profiler);

        const DepthOfFieldDilate::Output dilateOutput = m_dilatePass.record(
            scopeAlloc.child_scope(), cb,
            flattenOutput.tileMinMaxCircleOfConfusion, cam, nextFrame,
            profiler);

        gRenderResources.images->release(
            flattenOutput.tileMinMaxCircleOfConfusion);

        const DepthOfFieldGather::Input gatherInput{
            .halfResIllumination = setupOutput.halfResIllumination,
            .halfResCoC = setupOutput.halfResCircleOfConfusion,
            .dilatedTileMinMaxCoC = dilateOutput.dilatedTileMinMaxCoC,
        };
        const DepthOfFieldGather::Output fgGatherOutput = m_gatherPass.record(
            scopeAlloc.child_scope(), cb, gatherInput,
            DepthOfFieldGather::GatherType_Foreground, nextFrame, profiler);
        const DepthOfFieldGather::Output bgGatherOutput = m_gatherPass.record(
            scopeAlloc.child_scope(), cb, gatherInput,
            DepthOfFieldGather::GatherType_Background, nextFrame, profiler);

        const DepthOfFieldFilter::Output fgFilterOutput = m_filterPass.record(
            scopeAlloc.child_scope(), cb,
            fgGatherOutput.halfResBokehColorWeight, nextFrame,
            DepthOfFieldFilter::DebugNames{
                .scope = "  FilterFG",
                .outRes = "halfResFgColorWeightdFiltered",
            },
            profiler);
        gRenderResources.images->release(
            fgGatherOutput.halfResBokehColorWeight);
        const DepthOfFieldFilter::Output bgFilterOutput = m_filterPass.record(
            scopeAlloc.child_scope(), cb,
            bgGatherOutput.halfResBokehColorWeight, nextFrame,
            DepthOfFieldFilter::DebugNames{
                .scope = "  FilterBG",
                .outRes = "halfResBgColorWeightdFiltered",
            },
            profiler);
        gRenderResources.images->release(
            bgGatherOutput.halfResBokehColorWeight);

        ret = m_combinePass.record(
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

        gRenderResources.images->release(
            bgFilterOutput.filteredIlluminationWeight);
        gRenderResources.images->release(
            fgFilterOutput.filteredIlluminationWeight);
        gRenderResources.images->release(dilateOutput.dilatedTileMinMaxCoC);
        gRenderResources.images->release(setupOutput.halfResIllumination);
        gRenderResources.images->release(setupOutput.halfResCircleOfConfusion);
    }
    return ret;
}
