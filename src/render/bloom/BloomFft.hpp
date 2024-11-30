#ifndef PROSPER_RENDER_BLOOM_FFT_HPP
#define PROSPER_RENDER_BLOOM_FFT_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>

class BloomFft
{
  public:
    static constexpr vk::Format sFftFormat = vk::Format::eR32G32B32A32Sfloat;
    static constexpr uint32_t sMinResolution = 256;

    BloomFft() noexcept = default;
    ~BloomFft() = default;

    BloomFft(const BloomFft &other) = delete;
    BloomFft(BloomFft &&other) = delete;
    BloomFft &operator=(const BloomFft &other) = delete;
    BloomFft &operator=(BloomFft &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    void startFrame();

    // Inverse is unscaled, its values need to be divided by dim^2 when
    // interpreting
    [[nodiscard]] ImageHandle record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        ImageHandle input, uint32_t nextFrame, bool inverse,
        const char *debugPrefix);

  private:
    struct IterationData
    {
        ImageHandle input;
        ImageHandle output;
        uint32_t n{sMinResolution};
        uint32_t ns{1};
        uint32_t r{4};
        bool transpose{false};
        bool inverse{false};
    };
    void doIteration(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const IterationData &iterData, uint32_t nextFrame);

    bool m_initialized{false};
    ComputePass m_computePass;
};

#endif // PROSPER_RENDER_BLOOM_FFT_HPP
