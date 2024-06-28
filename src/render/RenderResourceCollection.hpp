#ifndef PROSPER_RENDER_RESOURCE_COLLECTION_HPP
#define PROSPER_RENDER_RESOURCE_COLLECTION_HPP

#include "../Allocators.hpp"
#include "../gfx/Device.hpp"
#include "../utils/Utils.hpp"
#include "RenderResourceHandle.hpp"

#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/static_array.hpp>

#include <vulkan/vulkan.hpp>

// TODO:
// This template signature is a mess. Is there a cleaner way without going full
// bonkers with macros.
template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
class RenderResourceCollection
{
  public:
    RenderResourceCollection() noexcept = default;
    virtual ~RenderResourceCollection();

    RenderResourceCollection(RenderResourceCollection &) = delete;
    RenderResourceCollection(RenderResourceCollection &&) = delete;
    RenderResourceCollection &operator=(RenderResourceCollection &) = delete;
    RenderResourceCollection &operator=(RenderResourceCollection &&) = delete;

    void startFrame();
    virtual void destroyResources();

    [[nodiscard]] Handle create(const Description &desc, const char *debugName);
    // Caller is expected to check validity before calling methods with the
    // handle. This design assumes that the code that creates and releases
    // resources is single-threaded and the handle isn't be released between
    // isValidHandle() and following accessor calls.
    [[nodiscard]] bool isValidHandle(Handle handle) const;
    [[nodiscard]] CppNativeType nativeHandle(Handle handle) const;
    [[nodiscard]] const Resource &resource(Handle handle) const;
    void transition(vk::CommandBuffer cb, Handle handle, ResourceState state);
    [[nodiscard]] wheels::Optional<Barrier> transitionBarrier(
        Handle handle, ResourceState state, bool force_barrier = false);
    void appendDebugName(Handle handle, wheels::StrSpan name);
    void preserve(Handle handle);
    void release(Handle handle);

    // Shouldn't be used by anything other than debug views, will only be valid
    // if the last aliased use for a resource. Marked debug resource will be
    // always valid.
    [[nodiscard]] wheels::Span<const wheels::String> debugNames() const;
    [[nodiscard]] Handle activeDebugHandle() const;
    [[nodiscard]] wheels::Optional<wheels::StrSpan> activeDebugName() const;
    void markForDebug(wheels::StrSpan debugName);
    void clearDebug();

  protected:
    void assertValidHandle(Handle handle) const;
    wheels::StrSpan aliasedDebugName(Handle handle) const;

  private:
    static const uint64_t sNotInUseGenerationFlag = static_cast<size_t>(1)
                                                    << 63;

    [[nodiscard]] bool resourceInUse(uint32_t i) const;
    void assertUniqueDebugName(wheels::StrSpan debugName) const;

    // RenderImageCollection depends on returned handle indices being
    // contiguous.
    wheels::Array<Resource> m_resources{gAllocators.general};
    wheels::Array<Description> m_descriptions{gAllocators.general};
    wheels::Array<wheels::String> m_aliasedDebugNames{gAllocators.general};
    wheels::Array<uint64_t> m_generations{gAllocators.general};
    wheels::Array<wheels::String> m_debugNames{gAllocators.general};
    wheels::Optional<wheels::String> m_markedDebugName;
    wheels::Optional<Handle> m_markedDebugHandle;
    wheels::Array<bool> m_preserved{gAllocators.general};
    wheels::Array<uint8_t> m_framesSinceUsed{gAllocators.general};
    // Indices of resource slots whose resource has been destroyed fully and so
    // the slot can be reused
    wheels::Array<uint32_t> m_freelist{gAllocators.general};
};

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::~RenderResourceCollection()
{
    RenderResourceCollection::destroyResources();
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
void RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::startFrame()
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
                !resourceInUse(asserted_cast<uint32_t>(i)) &&
                "Resource leaked");
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

                gDevice.destroy(m_resources[i]);
                m_resources[i] = Resource{};
                m_descriptions[i] = Description{};
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

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
void RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::destroyResources()
{
    for (Resource &res : m_resources)
        gDevice.destroy(res);

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

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
Handle RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType,
    ObjectType>::create(const Description &desc, const char *debugName)
{
    const uint32_t descCount = asserted_cast<uint32_t>(m_descriptions.size());
    for (uint32_t i = 0; i < descCount; ++i)
    {
        if (!resourceInUse(i))
        {
            WHEELS_ASSERT(!m_preserved[i]);

            const Description &existingDesc = m_descriptions[i];
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

                const Handle handle{
                    .index = i,
                    .generation = m_generations[i],
                };

                appendDebugName(handle, debugName);

                return handle;
            }
        }
    }

    uint32_t index = 0xFFFFFFFF;
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
    WHEELS_ASSERT(m_resources[index].handle == CppNativeType{});

    m_resources[index] = gDevice.create(CreateInfo{
        .desc = desc,
        .debugName = debugName,
    });
    m_descriptions[index] = desc;
    m_aliasedDebugNames[index].extend(debugName);
    uint64_t &generation = m_generations[index];
    generation = generation & ~sNotInUseGenerationFlag;

    m_preserved[index] = false;
    m_framesSinceUsed[index] = 0;

    const Handle handle{
        .index = index,
        .generation = m_generations[index],
    };

    assertValidHandle(handle);

    appendDebugName(handle, debugName);

    if (m_markedDebugName.has_value() && debugName == *m_markedDebugName)
        m_markedDebugHandle = handle;

    return handle;
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
bool RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::isValidHandle(Handle handle) const
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
        // Handle generation matching means held generation isn't flagged unused
        if (handle.generation != m_generations[handle.index])
            return false;
    return true;
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
CppNativeType RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::nativeHandle(Handle handle) const
{
    assertValidHandle(handle);

    return m_resources[handle.index].handle;
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
const Resource &RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::resource(Handle handle) const
{
    assertValidHandle(handle);

    return m_resources[handle.index];
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
void RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::
    transition(vk::CommandBuffer cb, Handle handle, ResourceState state)
{
    assertValidHandle(handle);

    m_resources[handle.index].transition(cb, state);
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
wheels::Optional<Barrier> RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::
    transitionBarrier(Handle handle, ResourceState state, bool force_barrier)
{
    assertValidHandle(handle);

    return m_resources[handle.index].transitionBarrier(state, force_barrier);
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
void RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType,
    ObjectType>::appendDebugName(Handle handle, wheels::StrSpan debugName)
{
    assertValidHandle(handle);

    wheels::String &aliasedName = m_aliasedDebugNames[handle.index];
    if (!aliasedName.empty())
        aliasedName.push_back('|');
    aliasedName.extend(debugName);

    // TODO: Set these at once? Need to be careful to set before
    // submits?
    gDevice.logical().setDebugUtilsObjectNameEXT(
        vk::DebugUtilsObjectNameInfoEXT{
            .objectType = ObjectType,
            .objectHandle = reinterpret_cast<uint64_t>(
                static_cast<NativeType>(m_resources[handle.index].handle)),
            .pObjectName = m_aliasedDebugNames[handle.index].c_str(),
        });

    assertUniqueDebugName(debugName);
    m_debugNames.emplace_back(gAllocators.general, debugName);

    if (m_markedDebugName.has_value() && debugName == *m_markedDebugName)
        m_markedDebugHandle = handle;
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
void RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::release(Handle handle)
{
    assertValidHandle(handle);

    // Releases on preserved resources are valid as no-ops so that the info
    // about preserving doesn't have to permeate the renderer.
    if (m_preserved[handle.index])
        return;

    m_generations[handle.index]++;
    m_generations[handle.index] |= sNotInUseGenerationFlag;
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
void RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::preserve(Handle handle)
{
    assertValidHandle(handle);
    WHEELS_ASSERT(
        !m_preserved[handle.index] &&
        "Resource is being preseved in two places, ownership gets muddy.");

    m_preserved[handle.index] = true;
    m_framesSinceUsed[handle.index] = 0;
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
wheels::Span<const wheels::String> RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::debugNames() const
{
    return m_debugNames;
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
Handle RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::activeDebugHandle() const
{
    if (!m_markedDebugHandle.has_value() ||
        !isValidHandle(*m_markedDebugHandle))
        return Handle{};

    return *m_markedDebugHandle;
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
wheels::Optional<wheels::StrSpan> RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::activeDebugName() const
{
    if (m_markedDebugName.has_value())
        return wheels::Optional<wheels::StrSpan>{*m_markedDebugName};

    return wheels::Optional<wheels::StrSpan>{};
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
void RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::markForDebug(wheels::StrSpan
                                                             debugName)
{
    m_markedDebugName = wheels::String{gAllocators.general, debugName};
    // Let's not worry about finding the resource immediately, we'll have it on
    // the next frame.
    m_markedDebugHandle.reset();
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
void RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::clearDebug()
{
    m_markedDebugName.reset();
    m_markedDebugHandle.reset();
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
void RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::assertValidHandle(Handle handle)
    const
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
#ifndef NDEBUG
        const uint64_t storedGeneration =
            m_generations[handle.index] & ~sNotInUseGenerationFlag;
        WHEELS_ASSERT(
            handle.generation == storedGeneration ||
            (handle.generation + 1) == storedGeneration);
#endif // NDEBUG
    }
    else
        // Handle generation matching means held generation isn't flagged unused
        WHEELS_ASSERT(handle.generation == m_generations[handle.index]);
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
wheels::StrSpan RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::aliasedDebugName(Handle handle)
    const
{
    WHEELS_ASSERT(isValidHandle(handle));
    return m_aliasedDebugNames[handle.index];
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
bool RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::resourceInUse(uint32_t i) const
{
    WHEELS_ASSERT(i < m_generations.size());
    return (m_generations[i] & sNotInUseGenerationFlag) == 0;
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
void RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType,
    ObjectType>::assertUniqueDebugName(wheels::StrSpan debugName) const
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

#endif // PROSPER_RENDER_RESOURCE_COLLECTION_HPP
