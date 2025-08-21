#ifndef PROSPER_RENDER_BLOOM_BLUR_HPP
#define PROSPER_RENDER_BLOOM_BLUR_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "render/bloom/ResolutionScale.hpp"

#include <glm/glm.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>

namespace render::bloom
{

class Blur
{
  public:
    Blur() noexcept = default;
    ~Blur() = default;

    Blur(const Blur &other) = delete;
    Blur(Blur &&other) = delete;
    Blur &operator=(const Blur &other) = delete;
    Blur &operator=(Blur &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void startFrame();

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    void record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        ImageHandle inOutHighlights, ResolutionScale resolutionScale,
        uint32_t nextFrame);

  private:
    struct SinglePassData
    {
        vk::DescriptorSet descriptorSet;
        uint32_t mipLevel{0};
        glm::uvec2 mipResolution;
        bool transpose{false};
    };
    void recordSinglePass(vk::CommandBuffer cb, const SinglePassData &data);

    bool m_initialized{false};
    ComputePass m_computePass;
};

} // namespace render::bloom

#endif // PROSPER_RENDER_BLOOM_BLUR_HPP
