#include "RenderImageCollection.hpp"

RenderImageCollection::~RenderImageCollection()
{
    RenderImageCollection::destroyResources();
}

void RenderImageCollection::destroyResources()
{
    for (auto &views : _subresourceViews)
    {
        gDevice.destroy(views);
        views.clear();
    }
    _subresourceViews.clear();

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
        gDevice.createSubresourcesViews(image, views.mut_span());
    }
    else
    {
        const Image &image = resource(handle);
        WHEELS_ASSERT(views.size() == image.subresourceRange.levelCount);
    }

    return views;
}
