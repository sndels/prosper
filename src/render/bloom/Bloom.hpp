#ifndef PROSPER_RENDER_BLOOM_HPP
#define PROSPER_RENDER_BLOOM_HPP

#include "render/RenderResourceHandle.hpp"
#include "render/bloom/BloomBlur.hpp"
#include "render/bloom/BloomCompose.hpp"
#include "render/bloom/BloomConvolution.hpp"
#include "render/bloom/BloomFft.hpp"
#include "render/bloom/BloomGenerateKernel.hpp"
#include "render/bloom/BloomReduce.hpp"
#include "render/bloom/BloomResolutionScale.hpp"
#include "render/bloom/BloomSeparate.hpp"
#include "render/bloom/BloomTechnique.hpp"

#include <filesystem>
#include <vulkan/vulkan.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/hash_set.hpp>

namespace render::bloom
{

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

    using Input = BloomSeparate::Input;
    struct Output
    {
        ImageHandle illuminationWithBloom;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const Input &input, uint32_t nextFrame);
    void releasePreserved();

  private:
    bool m_initialized{false};
    BloomResolutionScale m_resolutionScale{BloomResolutionScale::Half};
    BloomTechnique m_technique{BloomTechnique::MultiResolutionBlur};

    BloomSeparate m_separate;
    BloomCompose m_compose;

    // FFT version
    BloomGenerateKernel m_generateKernel;
    BloomFft m_fft;
    BloomConvolution m_convolution;

    // Multi-resolution blur version
    BloomReduce m_reduce;
    BloomBlur m_blur;
};

} // namespace render::bloom

#endif // PROSPER_RENDER_BLOOM_HPP
