#include "Bloom.hpp"

#include "render/RenderResources.hpp"
#include "render/Utils.hpp"
#include "utils/Profiler.hpp"

#include <imgui.h>

using namespace wheels;

void Bloom::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_separate.init(scopeAlloc.child_scope());
    m_generateKernel.init(scopeAlloc.child_scope());
    m_fft.init(scopeAlloc.child_scope());
    m_convolution.init(scopeAlloc.child_scope());
    m_compose.init(WHEELS_MOV(scopeAlloc));

    m_initialized = true;
}

void Bloom::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_separate.recompileShaders(scopeAlloc.child_scope(), changedFiles);
    m_generateKernel.recompileShaders(scopeAlloc.child_scope(), changedFiles);
    m_fft.recompileShaders(scopeAlloc.child_scope(), changedFiles);
    m_convolution.recompileShaders(scopeAlloc.child_scope(), changedFiles);
    m_compose.recompileShaders(WHEELS_MOV(scopeAlloc), changedFiles);
}

void Bloom::startFrame() { m_fft.startFrame(); }

void Bloom::drawUi()
{
    ImGui::Indent();
    m_separate.drawUi();
    m_generateKernel.drawUi();
    ImGui::Unindent();
}

Bloom::Output Bloom::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Input &input,
    const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_GPU_SCOPE(cb, "Bloom");

    const vk::Extent2D inputExtent = getExtent2D(input.illumination);
    ImageHandle kernelDft = m_generateKernel.record(
        scopeAlloc.child_scope(), cb, inputExtent, m_fft, nextFrame);

    ImageHandle workingImage =
        m_separate.record(scopeAlloc.child_scope(), cb, input, nextFrame);

    m_fft.record(
        scopeAlloc.child_scope(), cb, workingImage, nextFrame, false, "Bloom");

    m_convolution.record(
        scopeAlloc.child_scope(), cb,
        BloomConvolution::InputOutput{
            .inOutHighlightsDft = workingImage,
            .inKernelDft = kernelDft,
        },
        nextFrame);

    m_fft.record(
        scopeAlloc.child_scope(), cb, workingImage, nextFrame, true, "Bloom");

    const ImageHandle illuminationWithBloom = m_compose.record(
        WHEELS_MOV(scopeAlloc), cb,
        BloomCompose::Input{
            .illumination = input.illumination,
            .bloomHighlights = workingImage,
        },
        nextFrame);

    gRenderResources.images->release(workingImage);

    Output ret{
        .illuminationWithBloom = illuminationWithBloom,
    };

    return ret;
}

void Bloom::releasePreserved()
{
    WHEELS_ASSERT(m_initialized);

    m_generateKernel.releasePreserved();
}
