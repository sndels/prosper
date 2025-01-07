#include "Bloom.hpp"

#include "render/RenderResources.hpp"
#include "render/Utils.hpp"
#include "utils/Profiler.hpp"
#include "utils/Ui.hpp"

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
    enumDropdown(
        "Resolution scale", m_resolutionScale, sResolutionScaleTypeNames);
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
    const ImageHandle kernelDft = m_generateKernel.record(
        scopeAlloc.child_scope(), cb, inputExtent, m_fft, m_resolutionScale,
        nextFrame);

    const ImageHandle workingImage = m_separate.record(
        scopeAlloc.child_scope(), cb, input, m_resolutionScale, nextFrame);

    const ImageHandle highlightsDft = m_fft.record(
        scopeAlloc.child_scope(), cb, workingImage, nextFrame, false, "Bloom");

    gRenderResources.images->release(workingImage);

    float convolutionScale = m_generateKernel.convolutionScale();
    if (m_resolutionScale == BloomResolutionScale::Quarter)
        // This seems to match bloom intensity between quarter and half res
        convolutionScale *= 2.;

    m_convolution.record(
        scopeAlloc.child_scope(), cb,
        BloomConvolution::InputOutput{
            .inOutHighlightsDft = highlightsDft,
            .inKernelDft = kernelDft,
            .convolutionScale = convolutionScale,
        },
        nextFrame);

    const ImageHandle convolvedHighlights = m_fft.record(
        scopeAlloc.child_scope(), cb, highlightsDft, nextFrame, true, "Bloom");

    gRenderResources.images->release(highlightsDft);

    const ImageHandle illuminationWithBloom = m_compose.record(
        WHEELS_MOV(scopeAlloc), cb,
        BloomCompose::Input{
            .illumination = input.illumination,
            .bloomHighlights = convolvedHighlights,
        },
        m_resolutionScale, nextFrame);

    gRenderResources.images->release(convolvedHighlights);

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
