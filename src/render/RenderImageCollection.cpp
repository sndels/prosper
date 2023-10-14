#include "RenderImageCollection.hpp"

RenderImageCollection::RenderImageCollection(
    wheels::Allocator &alloc, Device *device)
: RenderResourceCollection{alloc, device}
, _subresourceViews{alloc}
{
}

RenderImageCollection::~RenderImageCollection()
{
    RenderImageCollection::destroyResources();
}

void RenderImageCollection::destroyResources()
{
    for (auto &views : _subresourceViews)
    {
        _device->destroy(views);
        views.clear();
    }

    RenderResourceCollection::destroyResources();
}

wheels::Span<const vk::ImageView> RenderImageCollection::subresourceViews(
    ImageHandle handle)
{
    assertValidHandle(handle);
    if (_subresourceViews.size() <= handle.index)
        _subresourceViews.resize(handle.index + 1);

    auto &views = _subresourceViews[handle.index];

    if (views.empty())
    {
        const Image &image = resource(handle);
        views.resize(image.subresourceRange.levelCount);
        _device->createSubresourcesViews(image, views);
    }
    else
    {
        const Image &image = resource(handle);
        WHEELS_ASSERT(views.size() == image.subresourceRange.levelCount);
    }

    return views;
}
