#include "RenderImageCollection.hpp"

#include "gfx/Device.hpp"
#include "utils/Utils.hpp"

using namespace wheels;

namespace render
{

RenderImageCollection::~RenderImageCollection()
{
    RenderImageCollection::destroyResources();
}

void RenderImageCollection::startFrame()
{
    const size_t resourceCount = m_resources.size();
    WHEELS_ASSERT(resourceCount == m_preserved.size());
    WHEELS_ASSERT(resourceCount == m_aliasedDebugNames.size());
    for (size_t i = 0; i < resourceCount; ++i)
    {
        // Get name for debug convenience
        const wheels::String &aliasedDebugName = m_aliasedDebugNames[i];
        if (m_preserved[i])
            m_preserved[i] = false;
        else
            WHEELS_ASSERT(
                !resourceInUse(asserted_cast<uint32_t>(i)) && "Image leaked");
        (void)aliasedDebugName;
    }

    // These are mapped to persistent resource indices
    for (wheels::String &str : m_aliasedDebugNames)
        str.clear();

    // These are collected each frame for every created resource
    for (wheels::String &str : m_debugNames)
        str.clear();
    m_debugNames.clear();

    WHEELS_ASSERT(resourceCount == m_framesSinceUsed.size());
    // This seems like a sufficiently conservative bound to avoid pingpong
    // destroys for resources that are needed on some frames
    const uint8_t destroyDelayFrames =
        asserted_cast<uint8_t>(2 * MAX_FRAMES_IN_FLIGHT);
    static_assert(
        destroyDelayFrames < 0xFF, "0xFF is marks destroyed resources");
    for (uint32_t i = 0; i < resourceCount; ++i)
    {
        uint8_t &unusedFrames = m_framesSinceUsed[i];
        if (unusedFrames < 0xFF)
        {
            if (unusedFrames > destroyDelayFrames)
            {
                WHEELS_ASSERT(!m_preserved[i]);

                gfx::gDevice.destroy(m_resources[i]);
                m_resources[i] = gfx::Image{};
                m_descriptions[i] = gfx::ImageDescription{};
                if (i < m_subresourceViews.size())
                {
                    gfx::gDevice.destroy(m_subresourceViews[i]);
                    m_subresourceViews[i].clear();
                }
                m_aliasedDebugNames[i].clear();
                // Generations should stay as is, we can reuse the handle for
                // another resource
                // Mark destroyed resource
                unusedFrames = 0xFF;
                m_freelist.push_back(i);
            }
            else
                unusedFrames++;
        }
    }

    m_markedDebugHandle.reset();
}

void RenderImageCollection::destroyResources()
{
    for (auto &views : m_subresourceViews)
    {
        gfx::gDevice.destroy(views);
        views.clear();
    }
    m_subresourceViews.clear();

    for (gfx::Image &res : m_resources)
        gfx::gDevice.destroy(res);

    m_resources.clear();
    m_descriptions.clear();
    m_aliasedDebugNames.clear();
    // Bump all generations to invalidate any stored handles
    for (uint64_t &generation : m_generations)
    {
        const uint64_t storedGeneration = generation & ~sNotInUseGenerationFlag;
        generation = sNotInUseGenerationFlag | (storedGeneration + 1);
    }
    m_debugNames.clear();
    // m_markedDebugName should be persistent and only cleared through an
    // explicit call to clearDebug()
    m_markedDebugHandle.reset();
    m_preserved.clear();
    m_framesSinceUsed.clear();
    m_freelist.clear();
}

ImageHandle RenderImageCollection::create(
    const gfx::ImageDescription &desc, const char *debugName)
{
    const uint32_t descCount = asserted_cast<uint32_t>(m_descriptions.size());
    for (uint32_t i = 0; i < descCount; ++i)
    {
        if (!resourceInUse(i))
        {
            WHEELS_ASSERT(!m_preserved[i]);

            const gfx::ImageDescription &existingDesc = m_descriptions[i];
            if (existingDesc.matches(desc))
            {
                // Don't reuse the actively debugged resource to avoid stomping
                // it
                if (m_markedDebugName.has_value() &&
                    m_aliasedDebugNames[i].ends_with(*m_markedDebugName))
                {
                    // Make sure we're not just partially matching the last part
                    // of the concatenated debug identifier
                    const size_t breakPosition = m_aliasedDebugNames[i].size() -
                                                 1 - m_markedDebugName->size();
                    if (m_aliasedDebugNames[i].size() ==
                            m_markedDebugName->size() ||
                        m_aliasedDebugNames[i][breakPosition] == '|')
                        continue;
                }

                m_generations[i] &= ~sNotInUseGenerationFlag;
                m_framesSinceUsed[i] = 0;

                const ImageHandle handle{
                    .index = i,
                    .generation = m_generations[i],
                };

                appendDebugName(handle, debugName);

                return handle;
            }
        }
    }

    uint32_t index = 0xFFFF'FFFF;
    if (!m_freelist.empty())
        index = m_freelist.pop_back();
    else
    {
        m_resources.emplace_back();
        m_descriptions.emplace_back();
        m_aliasedDebugNames.emplace_back(gAllocators.general);
        m_debugNames.emplace_back(gAllocators.general);
        m_preserved.push_back(false);
        m_framesSinceUsed.push_back((uint8_t)0);
        // We might have handle generations from previously destroyed resources
        if (m_generations.size() < m_resources.size())
        {
            m_generations.push_back((uint64_t)sNotInUseGenerationFlag);
        }
        index = asserted_cast<uint32_t>(m_resources.size() - 1);
    }
    WHEELS_ASSERT(!resourceInUse(index));
    WHEELS_ASSERT(m_resources[index].handle == vk::Image{});

    m_resources[index] = gfx::gDevice.create(gfx::ImageCreateInfo{
        .desc = desc,
        .debugName = debugName,
    });
    m_descriptions[index] = desc;
    m_aliasedDebugNames[index].extend(debugName);
    uint64_t &generation = m_generations[index];
    generation = generation & ~sNotInUseGenerationFlag;

    m_preserved[index] = false;
    m_framesSinceUsed[index] = 0;

    const ImageHandle handle{
        .index = index,
        .generation = m_generations[index],
    };

    assertValidHandle(handle);

    appendDebugName(handle, debugName);

    if (m_markedDebugName.has_value() && debugName == *m_markedDebugName)
        m_markedDebugHandle = handle;

    return handle;
}

bool RenderImageCollection::isValidHandle(ImageHandle handle) const
{
    // NOTE:
    // Any changes need to be mirrored in assertValidHandle().
    if (!handle.isValid())
        return false;
    if (handle.index >= m_resources.size())
        return false;
    if (handle.index >= m_generations.size())
        return false;
    if (m_markedDebugHandle.has_value() &&
        handle.index == m_markedDebugHandle->index)
    {
        const uint64_t storedGeneration =
            m_generations[handle.index] & ~sNotInUseGenerationFlag;
        if (handle.generation != storedGeneration &&
            (handle.generation + 1) != storedGeneration)
            return false;
    }
    else
        // ImageHandle generation matching means held generation isn't flagged
        // unused
        if (handle.generation != m_generations[handle.index])
            return false;
    return true;
}

vk::Image RenderImageCollection::nativeHandle(ImageHandle handle) const
{
    assertValidHandle(handle);

    return m_resources[handle.index].handle;
}

const gfx::Image &RenderImageCollection::resource(ImageHandle handle) const
{
    assertValidHandle(handle);

    return m_resources[handle.index];
}

Span<const vk::ImageView> RenderImageCollection::subresourceViews(
    ImageHandle handle)
{
    assertValidHandle(handle);
    if (m_subresourceViews.size() <= handle.index)
        m_subresourceViews.resize(handle.index + 1);

    const gfx::Image &image = resource(handle);
    // Let's be nice and return the single mip view for ergonomics in cases
    // where the logical resource might have one or many mips.
    if (image.mipCount == 1)
        return Span{&image.view, 1};

    InlineArray<vk::ImageView, sMaxMipCount> &views =
        m_subresourceViews[handle.index];
    if (views.empty())
    {
        views.resize(image.subresourceRange.levelCount);
        // TODO:
        // Isolate the last concatenated name if this gets shared resources at
        // some point? Is that always the 'active' logical resource?
        const StrSpan debugName = aliasedDebugName(handle);
        gfx::gDevice.createSubresourcesViews(
            image, debugName, views.mut_span());
    }
    WHEELS_ASSERT(views.size() == image.subresourceRange.levelCount);

    return views;
}

void RenderImageCollection::transition(
    vk::CommandBuffer cb, ImageHandle handle, gfx::ImageState state)
{
    assertValidHandle(handle);

    m_resources[handle.index].transition(cb, state);
}

wheels::Optional<vk::ImageMemoryBarrier2> RenderImageCollection::
    transitionBarrier(
        ImageHandle handle, gfx::ImageState state, bool force_barrier)
{
    assertValidHandle(handle);

    return m_resources[handle.index].transitionBarrier(state, force_barrier);
}

void RenderImageCollection::appendDebugName(
    ImageHandle handle, wheels::StrSpan debugName)
{
    assertValidHandle(handle);

    wheels::String &aliasedName = m_aliasedDebugNames[handle.index];
    if (!aliasedName.empty())
        aliasedName.push_back('|');
    aliasedName.extend(debugName);

    // TODO: Set these at once? Need to be careful to set before
    // submits?
    gfx::gDevice.logical().setDebugUtilsObjectNameEXT(
        vk::DebugUtilsObjectNameInfoEXT{
            .objectType = vk::ObjectType::eImage,
            .objectHandle = reinterpret_cast<uint64_t>(
                static_cast<VkImage>(m_resources[handle.index].handle)),
            .pObjectName = m_aliasedDebugNames[handle.index].c_str(),
        });

    gfx::gDevice.logical().setDebugUtilsObjectNameEXT(
        vk::DebugUtilsObjectNameInfoEXT{
            .objectType = vk::ObjectType::eImageView,
            .objectHandle = reinterpret_cast<uint64_t>(
                static_cast<VkImageView>(m_resources[handle.index].view)),
            .pObjectName = m_aliasedDebugNames[handle.index].c_str(),
        });

    assertUniqueDebugName(debugName);
    m_debugNames.emplace_back(gAllocators.general, debugName);

    if (m_markedDebugName.has_value() && debugName == *m_markedDebugName)
        m_markedDebugHandle = handle;
}

void RenderImageCollection::release(ImageHandle handle)
{
    assertValidHandle(handle);

    // Releases on preserved resources are valid as no-ops so that the info
    // about preserving doesn't have to permeate the renderer.
    if (m_preserved[handle.index])
        return;

    m_generations[handle.index]++;
    m_generations[handle.index] |= sNotInUseGenerationFlag;
}

void RenderImageCollection::preserve(ImageHandle handle)
{
    assertValidHandle(handle);
    WHEELS_ASSERT(
        !m_preserved[handle.index] &&
        "Image is being preseved in two places, ownership gets muddy.");

    m_preserved[handle.index] = true;
    m_framesSinceUsed[handle.index] = 0;
}

wheels::Span<const wheels::String> RenderImageCollection::debugNames() const
{
    return m_debugNames;
}

ImageHandle RenderImageCollection::activeDebugHandle() const
{
    if (!m_markedDebugHandle.has_value() ||
        !isValidHandle(*m_markedDebugHandle))
        return ImageHandle{};

    return *m_markedDebugHandle;
}

wheels::Optional<wheels::StrSpan> RenderImageCollection::activeDebugName() const
{
    if (m_markedDebugName.has_value())
        return wheels::Optional<wheels::StrSpan>{*m_markedDebugName};

    return wheels::Optional<wheels::StrSpan>{};
}

void RenderImageCollection::markForDebug(wheels::StrSpan debugName)
{
    m_markedDebugName = wheels::String{gAllocators.general, debugName};
    // Let's not worry about finding the resource immediately, we'll have it on
    // the next frame.
    m_markedDebugHandle.reset();
}

void RenderImageCollection::clearDebug()
{
    m_markedDebugName.reset();
    m_markedDebugHandle.reset();
}

void RenderImageCollection::assertValidHandle(ImageHandle handle) const
{
    // NOTE:
    // Any changes need to be mirrored in isValidHandle()!
    // Mirrored implementations so that this asserting version provides granular
    // info in a debugger
    WHEELS_ASSERT(handle.isValid());
    WHEELS_ASSERT(handle.index < m_resources.size());
    WHEELS_ASSERT(handle.index < m_generations.size());
    if (m_markedDebugHandle.has_value() &&
        handle.index == m_markedDebugHandle->index)
    {
        const uint64_t storedGeneration =
            m_generations[handle.index] & ~sNotInUseGenerationFlag;
        WHEELS_ASSERT(
            handle.generation == storedGeneration ||
            (handle.generation + 1) == storedGeneration);
    }
    else
        // ImageHandle generation matching means held generation isn't flagged
        // unused
        WHEELS_ASSERT(handle.generation == m_generations[handle.index]);
}

wheels::StrSpan RenderImageCollection::aliasedDebugName(
    ImageHandle handle) const
{
    WHEELS_ASSERT(isValidHandle(handle));
    return m_aliasedDebugNames[handle.index];
}

bool RenderImageCollection::resourceInUse(uint32_t i) const
{
    WHEELS_ASSERT(i < m_generations.size());
    return (m_generations[i] & sNotInUseGenerationFlag) == 0;
}

// Silence release build warning
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void RenderImageCollection::assertUniqueDebugName(
    wheels::StrSpan debugName) const
{
#ifndef NDEBUG
    for (const wheels::String &name : m_debugNames)
        WHEELS_ASSERT(
            debugName != name &&
            "Debug names need to be unique within a frame");
#else
    (void)debugName;
#endif // NDEBUG
}

} // namespace render
