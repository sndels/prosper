#ifndef PROSPER_RENDER_BLOOM_CONVOLUTION_HPP
#define PROSPER_RENDER_BLOOM_CONVOLUTION_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>

namespace render::bloom
{

class Convolution
{
  public:
    Convolution() noexcept = default;
    ~Convolution() = default;

    Convolution(const Convolution &other) = delete;
    Convolution(Convolution &&other) = delete;
    Convolution &operator=(const Convolution &other) = delete;
    Convolution &operator=(Convolution &&other) = delete;

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

} // namespace render::bloom

#endif // PROSPER_RENDER_BLOOM_CONVOLUTION_HPP
