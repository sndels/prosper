#ifndef PROSPER_RENDER_BLOOM_GENERATE_KERNEL_HPP
#define PROSPER_RENDER_BLOOM_GENERATE_KERNEL_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
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

    [[nodiscard]] ImageHandle record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const vk::Extent2D &renderExtent, BloomFft &fft, uint32_t nextFrame);
    void releasePreserved();

  private:
    bool m_initialized{false};
    bool m_reGenerate{false};
    ImageHandle m_kernelDft;
    ComputePass m_computePass;
};

#endif // PROSPER_RENDER_BLOOM_GENERATE_KERNEL_HPP
