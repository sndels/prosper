#ifndef PROSPER_RENDER_TEXEL_BUFFER_COLLECTION_HPP
#define PROSPER_RENDER_TEXEL_BUFFER_COLLECTION_HPP

#include "Allocators.hpp"
#include "gfx/Resources.hpp"
#include "render/RenderResourceHandle.hpp"

#include <vulkan/vulkan.hpp>
#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/static_array.hpp>
#include <wheels/containers/string.hpp>

namespace render
{

class RenderTexelBufferCollection
{
  public:
    RenderTexelBufferCollection() noexcept = default;
    virtual ~RenderTexelBufferCollection();

    RenderTexelBufferCollection(RenderTexelBufferCollection &) = delete;
    RenderTexelBufferCollection(RenderTexelBufferCollection &&) = delete;
    RenderTexelBufferCollection &operator=(RenderTexelBufferCollection &) =
        delete;
    RenderTexelBufferCollection &operator=(RenderTexelBufferCollection &&) =
        delete;

    void startFrame();
    void destroyResources();

    [[nodiscard]] TexelBufferHandle create(
        const gfx::TexelBufferDescription &desc, const char *debugName);
    // Caller is expected to check validity before calling methods with the
    // handle. This design assumes that the code that creates and releases
    // resources is single-threaded and the handle isn't be released between
    // isValidHandle() and following accessor calls.
    [[nodiscard]] bool isValidHandle(TexelBufferHandle handle) const;
    [[nodiscard]] vk::Buffer nativeHandle(TexelBufferHandle handle) const;
    [[nodiscard]] const gfx::TexelBuffer &resource(
        TexelBufferHandle handle) const;
    void transition(
        vk::CommandBuffer cb, TexelBufferHandle handle, gfx::BufferState state);
    [[nodiscard]] wheels::Optional<vk::BufferMemoryBarrier2> transitionBarrier(
        TexelBufferHandle handle, gfx::BufferState state,
        bool force_barrier = false);
    void appendDebugName(TexelBufferHandle handle, wheels::StrSpan name);
    void preserve(TexelBufferHandle handle);
    void release(TexelBufferHandle handle);

    [[nodiscard]] wheels::Span<const wheels::String> debugNames() const;
    [[nodiscard]] TexelBufferHandle activeDebugHandle() const;
    [[nodiscard]] wheels::Optional<wheels::StrSpan> activeDebugName() const;
    void markForDebug(wheels::StrSpan debugName);
    void clearDebug();

  private:
    static const uint64_t sNotInUseGenerationFlag = static_cast<size_t>(1)
                                                    << 63;

    void assertValidHandle(TexelBufferHandle handle) const;
    [[nodiscard]] wheels::StrSpan aliasedDebugName(
        TexelBufferHandle handle) const;
    [[nodiscard]] bool resourceInUse(uint32_t i) const;
    void assertUniqueDebugName(wheels::StrSpan debugName) const;

    // RenderImageCollection depends on returned handle indices being
    // contiguous.
    wheels::Array<gfx::TexelBuffer> m_resources{gAllocators.general};
    wheels::Array<gfx::TexelBufferDescription> m_descriptions{
        gAllocators.general};
    wheels::Array<wheels::String> m_aliasedDebugNames{gAllocators.general};
    wheels::Array<uint64_t> m_generations{gAllocators.general};
    wheels::Array<wheels::String> m_debugNames{gAllocators.general};
    wheels::Optional<wheels::String> m_markedDebugName;
    wheels::Optional<TexelBufferHandle> m_markedDebugHandle;
    wheels::Array<bool> m_preserved{gAllocators.general};
    wheels::Array<uint8_t> m_framesSinceUsed{gAllocators.general};
    // Indices of resource slots whose resource has been destroyed fully and so
    // the slot can be reused
    wheels::Array<uint32_t> m_freelist{gAllocators.general};
};

} // namespace render

#endif // PROSPER_RENDER_TEXEL_BUFFER_COLLECTION_HPP
