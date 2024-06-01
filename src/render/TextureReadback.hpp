#ifndef PROSPER_RENDER_TEXTURE_READBACK_HPP
#define PROSPER_RENDER_TEXTURE_READBACK_HPP

#include <glm/glm.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>

#include "../gfx/Fwd.hpp"
#include "../gfx/Resources.hpp"
#include "../utils/Fwd.hpp"
#include "ComputePass.hpp"
#include "Fwd.hpp"
#include "RenderResourceHandle.hpp"

class TextureReadback
{
  public:
    TextureReadback() noexcept = default;
    ~TextureReadback();

    TextureReadback(const TextureReadback &other) = delete;
    TextureReadback(TextureReadback &&other) = delete;
    TextureReadback &operator=(const TextureReadback &other) = delete;
    TextureReadback &operator=(TextureReadback &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc);

    void startFrame();

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    // Call this to queue a readback. Only one is allowed to be in flight at a
    // time. Should be plenty as long as these are used for UI things.
    void record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        ImageHandle inTexture, glm::vec2 px, uint32_t nextFrame,
        Profiler *profiler);

    // Call this to get the result of the queued readback if it has finished.
    // Returns empty when the readback hasn't finished.
    wheels::Optional<glm::vec4> readback();

  private:
    bool _initialized{false};
    ComputePass _computePass;
    int32_t _framesUntilReady{-1};
    Buffer _buffer;
};

#endif // PROSPER_RENDER_TEXTURE_READBACK_HPP
