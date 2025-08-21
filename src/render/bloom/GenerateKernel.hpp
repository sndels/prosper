#ifndef PROSPER_RENDER_BLOOM_GENERATE_KERNEL_HPP
#define PROSPER_RENDER_BLOOM_GENERATE_KERNEL_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "render/bloom/Fwd.hpp"
#include "render/bloom/ResolutionScale.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>

namespace render::bloom
{

class GenerateKernel
{
  public:
    GenerateKernel() noexcept = default;
    ~GenerateKernel() = default;

    GenerateKernel(const GenerateKernel &other) = delete;
    GenerateKernel(GenerateKernel &&other) = delete;
    GenerateKernel &operator=(const GenerateKernel &other) = delete;
    GenerateKernel &operator=(GenerateKernel &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    void drawUi();
    [[nodiscard]] float convolutionScale() const;

    [[nodiscard]] ImageHandle record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const vk::Extent2D &renderExtent, Fft &fft,
        ResolutionScale resolutionScale, uint32_t nextFrame);
    void releasePreserved();

  private:
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters) private
    [[nodiscard]] ImageHandle recordGenerate(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb, uint32_t dim,
        uint32_t nextFrame);

    void recordPrepare(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb, uint32_t dim,
        Fft &fft, ImageHandle inKernel, uint32_t nextFrame);

    bool m_initialized{false};
    bool m_reGenerate{false};
    ImageHandle m_kernelDft;
    uint32_t m_previousKernelImageDim{0};
    ComputePass m_generatePass;
    ComputePass m_preparePass;
};

} // namespace render::bloom

#endif // PROSPER_RENDER_BLOOM_GENERATE_KERNEL_HPP
