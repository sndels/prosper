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
    void record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        ImageHandle &inputOutput, uint32_t nextFrame, bool inverse,
        const char *debugPrefix);

  private:
    struct DispatchData
    {
        wheels::StaticArray<ImageHandle, 2> images;
        uint32_t n{sMinResolution};
        bool transpose{false};
        bool inverse{false};
        bool needsRadix2{false};
    };
    void dispatch(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const DispatchData &dispatchData, uint32_t nextFrame);

    bool m_initialized{false};
    ComputePass m_computePass;
};

#endif // PROSPER_RENDER_BLOOM_FFT_HPP
