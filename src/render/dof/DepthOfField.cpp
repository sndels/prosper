#include "DepthOfField.hpp"

#include "render/RenderResources.hpp"
#include "utils/Profiler.hpp"

using namespace wheels;

namespace render::dof
{

void DepthOfField::init(
    ScopedScratch scopeAlloc, vk::DescriptorSetLayout cameraDsLayout)
{
    WHEELS_ASSERT(!m_initialized);

    m_setupPass.init(scopeAlloc.child_scope(), cameraDsLayout);
    m_reducePass.init(scopeAlloc.child_scope());
    m_flattenPass.init(scopeAlloc.child_scope());
    m_dilatePass.init(scopeAlloc.child_scope());
    m_gatherPass.init(scopeAlloc.child_scope());
    m_filterPass.init(scopeAlloc.child_scope());
    m_combinePass.init(scopeAlloc.child_scope());

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

Output DepthOfField::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const scene::Camera &cam,
    const Input &input, uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_GPU_SCOPE(cb, "DepthOfField");

    Output ret;
    {

        const Setup::Output setupOutput = m_setupPass.record(
            scopeAlloc.child_scope(), cb, cam, input, nextFrame);

        m_reducePass.record(
            scopeAlloc.child_scope(), cb, setupOutput.halfResIllumination,
            nextFrame);

        const Flatten::Output flattenOutput = m_flattenPass.record(
            scopeAlloc.child_scope(), cb, setupOutput.halfResCircleOfConfusion,
            nextFrame);

        const Dilate::Output dilateOutput = m_dilatePass.record(
            scopeAlloc.child_scope(), cb,
            flattenOutput.tileMinMaxCircleOfConfusion, cam, nextFrame);

        gRenderResources.images->release(
            flattenOutput.tileMinMaxCircleOfConfusion);

        const Gather::Input gatherInput{
            .halfResIllumination = setupOutput.halfResIllumination,
            .halfResCoC = setupOutput.halfResCircleOfConfusion,
            .dilatedTileMinMaxCoC = dilateOutput.dilatedTileMinMaxCoC,
        };
        const Gather::Output fgGatherOutput = m_gatherPass.record(
            scopeAlloc.child_scope(), cb, gatherInput,
            Gather::GatherType_Foreground, nextFrame);
        const Gather::Output bgGatherOutput = m_gatherPass.record(
            scopeAlloc.child_scope(), cb, gatherInput,
            Gather::GatherType_Background, nextFrame);

        const Filter::Output fgFilterOutput = m_filterPass.record(
            scopeAlloc.child_scope(), cb,
            fgGatherOutput.halfResBokehColorWeight, nextFrame,
            Filter::DebugNames{
                .scope = "  FilterFG",
                .outRes = "halfResFgColorWeightdFiltered",
            });
        gRenderResources.images->release(
            fgGatherOutput.halfResBokehColorWeight);
        const Filter::Output bgFilterOutput = m_filterPass.record(
            scopeAlloc.child_scope(), cb,
            bgGatherOutput.halfResBokehColorWeight, nextFrame,
            Filter::DebugNames{
                .scope = "  FilterBG",
                .outRes = "halfResBgColorWeightdFiltered",
            });
        gRenderResources.images->release(
            bgGatherOutput.halfResBokehColorWeight);

        ret = m_combinePass.record(
            scopeAlloc.child_scope(), cb,
            Combine::Input{
                .halfResFgBokehWeight =
                    fgFilterOutput.filteredIlluminationWeight,
                .halfResBgBokehWeight =
                    bgFilterOutput.filteredIlluminationWeight,
                .halfResCircleOfConfusion =
                    setupOutput.halfResCircleOfConfusion,
                .illumination = input.illumination,
            },
            nextFrame);

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

} // namespace render::dof
