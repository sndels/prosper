#ifndef PROSPER_RENDER_RESOURCE_HANDLE_HPP
#define PROSPER_RENDER_RESOURCE_HANDLE_HPP

#include "gfx/Fwd.hpp"

#include <wheels/containers/pair.hpp>

namespace render
{

// Generation is designed to get incremented each time a handle is released,
// potentially multiple times per frame. A maximum of e.g. 256 generations
// would enough within a frame but we should also assert against using stale
// handles from previous frames. Hence uint64_t.
// TODO:
// Handle stale handle validation with less space? Have a wrapping generation of
// sufficient size to assume matching gen is actually the same gen?
template <typename Resource> struct RenderResourceHandle
{
  public:
    static const uint32_t sNullIndex = 0xFFFF'FFFF;

    // TODO:
    // Protect these? Clang-tidy doesn't like similar value ctor arguments,
    // friending collection messes up the template signature of this handle as
    // well.
    uint32_t index{sNullIndex};
    uint64_t generation{0};

    [[nodiscard]] bool isValid() const { return index != sNullIndex; }
};

using BufferHandle = RenderResourceHandle<gfx::Buffer>;
using TexelBufferHandle = RenderResourceHandle<gfx::TexelBuffer>;
using ImageHandle = RenderResourceHandle<gfx::Image>;

} // namespace render

#endif // PROSPER_RENDER_RESOURCE_HANDLE_HPP
