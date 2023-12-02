#ifndef PROSPER_RENDER_IMAGE_COLLECTION_HPP
#define PROSPER_RENDER_IMAGE_COLLECTION_HPP

#include "../gfx/Fwd.hpp"
#include "../utils/Utils.hpp"
#include "RenderResourceCollection.hpp"
#include "RenderResourceHandle.hpp"

#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/static_array.hpp>

#include <vulkan/vulkan.hpp>

class RenderImageCollection
: public RenderResourceCollection<
      ImageHandle, Image, ImageDescription, ImageCreateInfo, ImageState,
      vk::ImageMemoryBarrier2, vk::Image, VkImage, vk::ObjectType::eImage>
{
  public:
    RenderImageCollection(wheels::Allocator &alloc, Device *device);
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
    wheels::Array<wheels::StaticArray<vk::ImageView, sMaxMipCount>>
        _subresourceViews;
};

#endif // PROSPER_RENDER_IMAGE_COLLECTION_HPP
