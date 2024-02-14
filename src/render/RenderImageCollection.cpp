#include "RenderImageCollection.hpp"

RenderImageCollection::RenderImageCollection(wheels::Allocator &alloc) noexcept
: RenderResourceCollection{alloc}
, _subresourceViews{alloc}
{
}

RenderImageCollection::~RenderImageCollection()
{
    RenderImageCollection::destroyResources();
}

void RenderImageCollection::destroyResources()
{
    if (_device != nullptr)
    {
        for (auto &views : _subresourceViews)
        {
            _device->destroy(views);
            views.clear();
        }
    }

    RenderResourceCollection::destroyResources();
}

wheels::Span<const vk::ImageView> RenderImageCollection::subresourceViews(
    ImageHandle handle)
{
    WHEELS_ASSERT(_device != nullptr);

    assertValidHandle(handle);
    if (_subresourceViews.size() <= handle.index)
        _subresourceViews.resize(handle.index + 1);

    auto &views = _subresourceViews[handle.index];

    if (views.empty())
    {
        const Image &image = resource(handle);
        views.resize(image.subresourceRange.levelCount);
        _device->createSubresourcesViews(image, views.mut_span());
    }
    else
    {
        const Image &image = resource(handle);
        WHEELS_ASSERT(views.size() == image.subresourceRange.levelCount);
    }

    return views;
}
