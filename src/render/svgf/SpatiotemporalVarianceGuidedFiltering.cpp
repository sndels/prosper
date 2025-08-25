#include "SpatiotemporalVarianceGuidedFiltering.hpp"

#include "render/RenderResources.hpp"
#include "utils/Profiler.hpp"

#include <imgui.h>

using namespace wheels;

namespace render::svgf
{

void SpatiotemporalVarianceGuidedFiltering::init(
    ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDSLayout)
{
    WHEELS_ASSERT(!m_initialized);

    m_accumulate.init(scopeAlloc.child_scope(), camDSLayout);

    m_initialized = true;
}

void SpatiotemporalVarianceGuidedFiltering::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    vk::DescriptorSetLayout camDSLayout)
{
    WHEELS_ASSERT(m_initialized);

    m_ignoreHistory |= m_accumulate.recompileShaders(
        scopeAlloc.child_scope(), changedFiles, camDSLayout);
}

void SpatiotemporalVarianceGuidedFiltering::drawUi()
{
    WHEELS_ASSERT(m_initialized);

    m_ignoreHistory |= ImGui::Button("Reset accumulation");
}

void SpatiotemporalVarianceGuidedFiltering::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const scene::Camera &cam,
    const Input &input, uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_GPU_SCOPE(cb, "SVGF");

    // TODO:
    // Ignore history on camera cuts

    {
        const Accumulate::Output accumulateOutput = m_accumulate.record(
            scopeAlloc.child_scope(), cb, cam, input, m_ignoreHistory,
            nextFrame);

        gRenderResources.images->release(accumulateOutput.color);
        gRenderResources.images->release(accumulateOutput.moments);
    }

    m_ignoreHistory = false;
}

void SpatiotemporalVarianceGuidedFiltering::releasePreserved()
{
    WHEELS_ASSERT(m_initialized);

    m_accumulate.releasePreserved();
}

} // namespace render::svgf
