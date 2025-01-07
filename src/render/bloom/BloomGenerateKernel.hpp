#ifndef PROSPER_RENDER_BLOOM_GENERATE_KERNEL_HPP
#define PROSPER_RENDER_BLOOM_GENERATE_KERNEL_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "render/bloom/BloomResolutionScale.hpp"
#include "render/bloom/Fwd.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>

class BloomGenerateKernel
{
  public:
    BloomGenerateKernel() noexcept = default;
    ~BloomGenerateKernel() = default;

    BloomGenerateKernel(const BloomGenerateKernel &other) = delete;
    BloomGenerateKernel(BloomGenerateKernel &&other) = delete;
    BloomGenerateKernel &operator=(const BloomGenerateKernel &other) = delete;
    BloomGenerateKernel &operator=(BloomGenerateKernel &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    void drawUi();
    [[nodiscard]] float convolutionScale() const;

    [[nodiscard]] ImageHandle record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const vk::Extent2D &renderExtent, BloomFft &fft,
        BloomResolutionScale resolutionScale, uint32_t nextFrame);
    void releasePreserved();

  private:
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters) private
    [[nodiscard]] ImageHandle recordGenerate(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb, uint32_t dim,
        uint32_t nextFrame);

    void recordPrepare(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb, uint32_t dim,
        BloomFft &fft, ImageHandle inKernel, uint32_t nextFrame);

    bool m_initialized{false};
    bool m_reGenerate{false};
    ImageHandle m_kernelDft;
    uint32_t m_previousKernelImageDim{0};
    ComputePass m_generatePass;
    ComputePass m_preparePass;
};

#endif // PROSPER_RENDER_BLOOM_GENERATE_KERNEL_HPP
