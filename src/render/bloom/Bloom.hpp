#ifndef PROSPER_RENDER_BLOOM_HPP
#define PROSPER_RENDER_BLOOM_HPP

#include "render/RenderResourceHandle.hpp"
#include "render/bloom/Blur.hpp"
#include "render/bloom/Compose.hpp"
#include "render/bloom/Convolution.hpp"
#include "render/bloom/Fft.hpp"
#include "render/bloom/GenerateKernel.hpp"
#include "render/bloom/Reduce.hpp"
#include "render/bloom/ResolutionScale.hpp"
#include "render/bloom/Separate.hpp"
#include "render/bloom/Technique.hpp"

#include <filesystem>
#include <vulkan/vulkan.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/hash_set.hpp>

namespace render::bloom
{

using Input = Separate::Input;
struct Output
{
    ImageHandle illuminationWithBloom;
};

class Bloom
{
  public:
    Bloom() noexcept = default;
    ~Bloom() = default;

    Bloom(const Bloom &other) = delete;
    Bloom(Bloom &&other) = delete;
    Bloom &operator=(const Bloom &other) = delete;
    Bloom &operator=(Bloom &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    void startFrame();

    void drawUi();

    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const Input &input, uint32_t nextFrame);
    void releasePreserved();

  private:
    bool m_initialized{false};
    ResolutionScale m_resolutionScale{ResolutionScale::Half};
    Technique m_technique{Technique::MultiResolutionBlur};

    Separate m_separate;
    Compose m_compose;

    // FFT version
    GenerateKernel m_generateKernel;
    Fft m_fft;
    Convolution m_convolution;

    // Multi-resolution blur version
    Reduce m_reduce;
    Blur m_blur;
};

} // namespace render::bloom

#endif // PROSPER_RENDER_BLOOM_HPP
