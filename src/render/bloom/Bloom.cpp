#include "Bloom.hpp"

#include "render/RenderResources.hpp"
#include "utils/Profiler.hpp"

#include <imgui.h>

using namespace wheels;

void Bloom::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_separate.init(scopeAlloc.child_scope());
    m_fft.init(WHEELS_MOV(scopeAlloc));

    m_initialized = true;
}

void Bloom::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_separate.recompileShaders(scopeAlloc.child_scope(), changedFiles);
    m_fft.recompileShaders(WHEELS_MOV(scopeAlloc), changedFiles);
}

void Bloom::startFrame() { m_fft.startFrame(); }

void Bloom::drawUi()
{
    ImGui::Indent();
    m_separate.drawUi();
    ImGui::Unindent();
}

Bloom::Output Bloom::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Input &input,
    const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_GPU_SCOPE(cb, "Bloom");

    const BloomSeparate::Output separateOutput =
        m_separate.record(scopeAlloc.child_scope(), cb, input, nextFrame);

    const ImageHandle fftOutput = m_fft.record(
        WHEELS_MOV(scopeAlloc), cb, separateOutput.highlights, nextFrame,
        false);

    const ImageHandle iFftOutput =
        m_fft.record(WHEELS_MOV(scopeAlloc), cb, fftOutput, nextFrame, true);

    gRenderResources.images->release(fftOutput);
    gRenderResources.images->release(iFftOutput);

    Output ret{
        .illuminationWithBloom = separateOutput.highlights,
    };

    return ret;
}
