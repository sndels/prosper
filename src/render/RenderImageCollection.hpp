#ifndef PROSPER_RENDER_IMAGE_COLLECTION_HPP
#define PROSPER_RENDER_IMAGE_COLLECTION_HPP

#include "Allocators.hpp"
#include "gfx/Fwd.hpp"
#include "render/RenderResourceCollection.hpp"
#include "render/RenderResourceHandle.hpp"
#include "utils/Utils.hpp"

#include <vulkan/vulkan.hpp>
#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/inline_array.hpp>

class RenderImageCollection
: public RenderResourceCollection<
      ImageHandle, Image, ImageDescription, ImageCreateInfo, ImageState,
      vk::ImageMemoryBarrier2, vk::Image, VkImage, vk::ObjectType::eImage>
{
  public:
    RenderImageCollection() noexcept = default;
    ~RenderImageCollection() override;

    RenderImageCollection(RenderImageCollection &) = delete;
    RenderImageCollection(RenderImageCollection &&) = delete;
    RenderImageCollection &operator=(RenderImageCollection &) = delete;
    RenderImageCollection &operator=(RenderImageCollection &&) = delete;

    void destroyResources() override;

    wheels::Span<const vk::ImageView> subresourceViews(ImageHandle handle);

  private:
    // mips for a 16k by 16k image
    static const size_t sMaxMipCount = 16;

    // TODO:
    // Is the sparsity of this array a memory usage problem?
    wheels::Array<wheels::InlineArray<vk::ImageView, sMaxMipCount>>
        m_subresourceViews{gAllocators.general};
    wheels::Array<vk::Image> m_cachedImages{gAllocators.general};
};

#endif // PROSPER_RENDER_IMAGE_COLLECTION_HPP
