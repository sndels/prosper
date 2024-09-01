#ifndef PROSPER_RENDER_BUFFER_COLLECTION_HPP
#define PROSPER_RENDER_BUFFER_COLLECTION_HPP

#include "Allocators.hpp"
#include "gfx/Resources.hpp"
#include "render/RenderResourceHandle.hpp"

#include <vulkan/vulkan.hpp>
#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/static_array.hpp>
#include <wheels/containers/string.hpp>

class RenderBufferCollection
{
  public:
    RenderBufferCollection() noexcept = default;
    ~RenderBufferCollection();

    RenderBufferCollection(RenderBufferCollection &) = delete;
    RenderBufferCollection(RenderBufferCollection &&) = delete;
    RenderBufferCollection &operator=(RenderBufferCollection &) = delete;
    RenderBufferCollection &operator=(RenderBufferCollection &&) = delete;

    void startFrame();
    void destroyResources();

    [[nodiscard]] BufferHandle create(
        const BufferDescription &desc, const char *debugName);
    // Caller is expected to check validity before calling methods with the
    // handle. This design assumes that the code that creates and releases
    // resources is single-threaded and the handle isn't be released between
    // isValidHandle() and following accessor calls.
    [[nodiscard]] bool isValidHandle(BufferHandle handle) const;
    [[nodiscard]] vk::Buffer nativeHandle(BufferHandle handle) const;
    [[nodiscard]] const Buffer &resource(BufferHandle handle) const;
    void transition(
        vk::CommandBuffer cb, BufferHandle handle, BufferState state);
    [[nodiscard]] wheels::Optional<vk::BufferMemoryBarrier2> transitionBarrier(
        BufferHandle handle, BufferState state, bool force_barrier = false);
    void appendDebugName(BufferHandle handle, wheels::StrSpan name);
    void preserve(BufferHandle handle);
    void release(BufferHandle handle);

    // Shouldn't be used by anything other than debug views, will only be valid
    // if the last aliased use for a resource. Marked debug resource will be
    // always valid.
    [[nodiscard]] wheels::Span<const wheels::String> debugNames() const;
    [[nodiscard]] BufferHandle activeDebugHandle() const;
    [[nodiscard]] wheels::Optional<wheels::StrSpan> activeDebugName() const;
    void markForDebug(wheels::StrSpan debugName);
    void clearDebug();

  private:
    static const uint64_t sNotInUseGenerationFlag = static_cast<size_t>(1)
                                                    << 63;

    void assertValidHandle(BufferHandle handle) const;
    [[nodiscard]] wheels::StrSpan aliasedDebugName(BufferHandle handle) const;
    [[nodiscard]] bool resourceInUse(uint32_t i) const;
    void assertUniqueDebugName(wheels::StrSpan debugName) const;

    wheels::Array<Buffer> m_resources{gAllocators.general};
    wheels::Array<BufferDescription> m_descriptions{gAllocators.general};
    wheels::Array<wheels::String> m_aliasedDebugNames{gAllocators.general};
    wheels::Array<uint64_t> m_generations{gAllocators.general};
    wheels::Array<wheels::String> m_debugNames{gAllocators.general};
    wheels::Optional<wheels::String> m_markedDebugName;
    wheels::Optional<BufferHandle> m_markedDebugHandle;
    wheels::Array<bool> m_preserved{gAllocators.general};
    wheels::Array<uint8_t> m_framesSinceUsed{gAllocators.general};
    // Indices of resource slots whose resource has been destroyed fully and so
    // the slot can be reused
    wheels::Array<uint32_t> m_freelist{gAllocators.general};
};

#endif // PROSPER_RENDER_BUFFER_COLLECTION_HPP
