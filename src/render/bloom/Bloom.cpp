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

    ImageHandle workingImage =
        m_separate.record(scopeAlloc.child_scope(), cb, input, nextFrame);

    const ImageHandle fftOutput = m_fft.record(
        WHEELS_MOV(scopeAlloc), cb, workingImage, nextFrame, false, "Bloom");

    gRenderResources.images->release(workingImage);

    const ImageHandle iFftOutput = m_fft.record(
        WHEELS_MOV(scopeAlloc), cb, fftOutput, nextFrame, true, "Bloom");

    gRenderResources.images->release(fftOutput);

    Output ret{
        .illuminationWithBloom = iFftOutput,
    };

    return ret;
}
