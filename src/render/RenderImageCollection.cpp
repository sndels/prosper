#include "RenderImageCollection.hpp"

using namespace wheels;

RenderImageCollection::~RenderImageCollection()
{
    RenderImageCollection::destroyResources();
}

void RenderImageCollection::destroyResources()
{
    for (auto &views : m_subresourceViews)
    {
        gDevice.destroy(views);
        views.clear();
    }
    m_subresourceViews.clear();

    RenderResourceCollection::destroyResources();
}

wheels::Span<const vk::ImageView> RenderImageCollection::subresourceViews(
    ImageHandle handle)
{
    assertValidHandle(handle);
    if (m_subresourceViews.size() <= handle.index)
        m_subresourceViews.resize(handle.index + 1);

    auto &views = m_subresourceViews[handle.index];

    if (views.empty())
    {
        const Image &image = resource(handle);
        // Let's be nice and return the single mip view for ergonomics in cases
        // where the logical resource might have one or many mips.
        if (image.mipCount == 1)
            return Span{&image.view, 1};

        views.resize(image.subresourceRange.levelCount);
        // TODO:
        // Isolate the last concatenated name if this gets shared resources at
        // some point? Is that always the 'active' logical resource?
        const StrSpan debugName = aliasedDebugName(handle);
        gDevice.createSubresourcesViews(image, debugName, views.mut_span());
    }
    else
    {
        const Image &image = resource(handle);
        WHEELS_ASSERT(views.size() == image.subresourceRange.levelCount);
    }

    return views;
}
