#ifndef PROSPER_RENDER_BLOOM_FFT_HPP
#define PROSPER_RENDER_BLOOM_FFT_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>

struct ComplexImagePair
{
    ImageHandle real;
    ImageHandle imag;
};

class BloomFft
{
  public:
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

    [[nodiscard]] ComplexImagePair record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const ComplexImagePair &input, uint32_t nextFrame, bool inverse);

  private:
    struct IterationData
    {
        ComplexImagePair input;
        ComplexImagePair output;
        uint32_t ns{1};
        uint32_t r{4};
        bool transpose{false};
    };
    void doIteration(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const IterationData &iterData, uint32_t nextFrame);

    bool m_initialized{false};
    ComputePass m_computePass;
};

#endif // PROSPER_RENDER_BLOOM_FFT_HPP
