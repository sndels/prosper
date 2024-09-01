#ifndef PROSPER_RENDER_IMAGE_COLLECTION_HPP
#define PROSPER_RENDER_IMAGE_COLLECTION_HPP

#include "Allocators.hpp"
#include "gfx/Resources.hpp"
#include "render/RenderResourceHandle.hpp"

#include <vulkan/vulkan.hpp>
#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/inline_array.hpp>
#include <wheels/containers/string.hpp>

class RenderImageCollection
{
  public:
    RenderImageCollection() noexcept = default;
    ~RenderImageCollection();

    RenderImageCollection(RenderImageCollection &) = delete;
    RenderImageCollection(RenderImageCollection &&) = delete;
    RenderImageCollection &operator=(RenderImageCollection &) = delete;
    RenderImageCollection &operator=(RenderImageCollection &&) = delete;

    void startFrame();
    void destroyResources();

    [[nodiscard]] ImageHandle create(
        const ImageDescription &desc, const char *debugName);
    // Caller is expected to check validity before calling methods with the
    // handle. This design assumes that the code that creates and releases
    // resources is single-threaded and the handle isn't be released between
    // isValidHandle() and following accessor calls.
    [[nodiscard]] bool isValidHandle(ImageHandle handle) const;
    [[nodiscard]] vk::Image nativeHandle(ImageHandle handle) const;
    [[nodiscard]] const Image &resource(ImageHandle handle) const;
    [[nodiscard]] wheels::Span<const vk::ImageView> subresourceViews(
        ImageHandle handle);
    void transition(vk::CommandBuffer cb, ImageHandle handle, ImageState state);
    [[nodiscard]] wheels::Optional<vk::ImageMemoryBarrier2> transitionBarrier(
        ImageHandle handle, ImageState state, bool force_barrier = false);
    void appendDebugName(ImageHandle handle, wheels::StrSpan name);
    void preserve(ImageHandle handle);
    void release(ImageHandle handle);

    // Shouldn't be used by anything other than debug views, will only be valid
    // if the last aliased use for a resource. Marked debug resource will be
    // always valid.
    [[nodiscard]] wheels::Span<const wheels::String> debugNames() const;
    [[nodiscard]] ImageHandle activeDebugHandle() const;
    [[nodiscard]] wheels::Optional<wheels::StrSpan> activeDebugName() const;
    void markForDebug(wheels::StrSpan debugName);
    void clearDebug();

  private:
    // mips for a 16k by 16k image
    static const size_t sMaxMipCount = 16;
    static const uint64_t sNotInUseGenerationFlag = static_cast<size_t>(1)
                                                    << 63;

    void assertValidHandle(ImageHandle handle) const;
    [[nodiscard]] wheels::StrSpan aliasedDebugName(ImageHandle handle) const;
    [[nodiscard]] bool resourceInUse(uint32_t i) const;
    void assertUniqueDebugName(wheels::StrSpan debugName) const;

    wheels::Array<Image> m_resources{gAllocators.general};
    wheels::Array<ImageDescription> m_descriptions{gAllocators.general};
    // TODO:
    // Is the sparsity of this array a memory usage problem?
    wheels::Array<wheels::InlineArray<vk::ImageView, sMaxMipCount>>
        m_subresourceViews{gAllocators.general};
    wheels::Array<wheels::String> m_aliasedDebugNames{gAllocators.general};
    wheels::Array<uint64_t> m_generations{gAllocators.general};
    wheels::Array<wheels::String> m_debugNames{gAllocators.general};
    wheels::Optional<wheels::String> m_markedDebugName;
    wheels::Optional<ImageHandle> m_markedDebugHandle;
    wheels::Array<bool> m_preserved{gAllocators.general};
    wheels::Array<uint8_t> m_framesSinceUsed{gAllocators.general};
    // Indices of resource slots whose resource has been destroyed fully and so
    // the slot can be reused
    wheels::Array<uint32_t> m_freelist{gAllocators.general};
};

#endif // PROSPER_RENDER_IMAGE_COLLECTION_HPP
