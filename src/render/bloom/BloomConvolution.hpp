#ifndef PROSPER_RENDER_BLOOM_CONVOLUTION_HPP
#define PROSPER_RENDER_BLOOM_CONVOLUTION_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>

class BloomConvolution
{
  public:
    BloomConvolution() noexcept = default;
    ~BloomConvolution() = default;

    BloomConvolution(const BloomConvolution &other) = delete;
    BloomConvolution(BloomConvolution &&other) = delete;
    BloomConvolution &operator=(const BloomConvolution &other) = delete;
    BloomConvolution &operator=(BloomConvolution &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    struct InputOutput
    {
        ImageHandle inOutHighlightsDft;
        ImageHandle inKernelDft;
        float convolutionScale{0.f};
    };
    void record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const InputOutput &inputOutput, uint32_t nextFrame);

  private:
    bool m_initialized{false};
    ComputePass m_computePass;
};

#endif // PROSPER_RENDER_BLOOM_CONVOLUTION_HPP
