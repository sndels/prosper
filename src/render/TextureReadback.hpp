#ifndef PROSPER_RENDER_TEXTURE_READBACK_HPP
#define PROSPER_RENDER_TEXTURE_READBACK_HPP

#include "RenderResourceHandle.hpp"
#include "gfx/Fwd.hpp"
#include "gfx/Resources.hpp"
#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"

#include <glm/glm.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>

class TextureReadback
{
  public:
    TextureReadback() noexcept = default;
    ~TextureReadback();

    TextureReadback(const TextureReadback &other) = delete;
    TextureReadback(TextureReadback &&other) = delete;
    TextureReadback &operator=(const TextureReadback &other) = delete;
    TextureReadback &operator=(TextureReadback &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void startFrame();

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    // Call this to queue a readback. Only one is allowed to be in flight at a
    // time. Should be plenty as long as these are used for UI things.
    void record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        ImageHandle inTexture, glm::vec2 px, uint32_t nextFrame);

    // Call this to get the result of the queued readback if it has finished.
    // Returns empty when the readback hasn't finished.
    wheels::Optional<glm::vec4> readback();

  private:
    bool m_initialized{false};
    ComputePass m_computePass;
    int32_t m_framesUntilReady{-1};
    Buffer m_buffer;
};

#endif // PROSPER_RENDER_TEXTURE_READBACK_HPP
