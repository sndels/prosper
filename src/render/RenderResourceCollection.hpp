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

    void init(Device *device);

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
    Device *_device{nullptr};

    void assertValidHandle(Handle handle) const;

  private:
    static const uint64_t sNotInUseGenerationFlag = static_cast<size_t>(1)
                                                    << 63;

    [[nodiscard]] bool resourceInUse(uint32_t i) const;
    void assertUniqueDebugName(wheels::StrSpan debugName) const;

    // RenderImageCollection depends on returned handle indices being
    // contiguous.
    wheels::Array<Resource> _resources{gAllocators.general};
    wheels::Array<Description> _descriptions{gAllocators.general};
    wheels::Array<wheels::String> _aliasedDebugNames{gAllocators.general};
    wheels::Array<uint64_t> _generations{gAllocators.general};
    wheels::Array<wheels::String> _debugNames{gAllocators.general};
    wheels::Optional<wheels::String> _markedDebugName;
    wheels::Optional<Handle> _markedDebugHandle;
    wheels::Array<bool> _preserved{gAllocators.general};
    wheels::Array<uint8_t> _framesSinceUsed{gAllocators.general};
    // Indices of resource slots whose resource has been destroyed fully and so
    // the slot can be reused
    wheels::Array<uint32_t> _freelist{gAllocators.general};
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
    CppNativeType, NativeType, ObjectType>::init(Device *device)
{
    WHEELS_ASSERT(_device == nullptr);
    WHEELS_ASSERT(device != nullptr);

    _device = device;
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
void RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::startFrame()
{
    WHEELS_ASSERT(_device != nullptr);

    const size_t resourceCount = _resources.size();
    WHEELS_ASSERT(resourceCount == _preserved.size());
    WHEELS_ASSERT(resourceCount == _aliasedDebugNames.size());
    for (size_t i = 0; i < resourceCount; ++i)
    {
        // Get name for debug convenience
        const wheels::String &aliasedDebugName = _aliasedDebugNames[i];
        if (_preserved[i])
            _preserved[i] = false;
        else
            WHEELS_ASSERT(
                !resourceInUse(asserted_cast<uint32_t>(i)) &&
                "Resource leaked");
        (void)aliasedDebugName;
    }

    // These are mapped to persistent resource indices
    for (wheels::String &str : _aliasedDebugNames)
        str.clear();

    // These are collected each frame for every created resource
    for (wheels::String &str : _debugNames)
        str.clear();
    _debugNames.clear();

    WHEELS_ASSERT(resourceCount == _framesSinceUsed.size());
    // This seems like a sufficiently conservative bound to avoid pingpong
    // destroys for resources that are needed on some frames
    const uint8_t destroyDelayFrames =
        asserted_cast<uint8_t>(2 * MAX_FRAMES_IN_FLIGHT);
    static_assert(
        destroyDelayFrames < 0xFF, "0xFF is marks destroyed resources");
    for (uint32_t i = 0; i < resourceCount; ++i)
    {
        uint8_t &unusedFrames = _framesSinceUsed[i];
        if (unusedFrames < 0xFF)
        {
            if (unusedFrames > destroyDelayFrames)
            {
                WHEELS_ASSERT(!_preserved[i]);

                _device->destroy(_resources[i]);
                _resources[i] = Resource{};
                _descriptions[i] = Description{};
                _aliasedDebugNames[i].clear();
                // Generations should stay as is, we can reuse the handle for
                // another resource
                // Mark destroyed resource
                unusedFrames = 0xFF;
                _freelist.push_back(i);
            }
            else
                unusedFrames++;
        }
    }

    _markedDebugHandle.reset();
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
void RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::destroyResources()
{
    if (_device != nullptr)
    {
        for (Resource &res : _resources)
            _device->destroy(res);

        _resources.clear();
        _descriptions.clear();
        _aliasedDebugNames.clear();
        // Bump all generations to invalidate any stored handles
        for (uint64_t &generation : _generations)
        {
            const uint64_t storedGeneration =
                generation & ~sNotInUseGenerationFlag;
            generation = sNotInUseGenerationFlag | (storedGeneration + 1);
        }
        _debugNames.clear();
        // _markedDebugName should be persistent and only cleared through an
        // explicit call to clearDebug()
        _markedDebugHandle.reset();
        _preserved.clear();
        _framesSinceUsed.clear();
        _freelist.clear();
    }
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
    WHEELS_ASSERT(_device != nullptr);

    const uint32_t descCount = asserted_cast<uint32_t>(_descriptions.size());
    for (uint32_t i = 0; i < descCount; ++i)
    {
        if (!resourceInUse(i))
        {
            WHEELS_ASSERT(!_preserved[i]);

            const Description &existingDesc = _descriptions[i];
            if (existingDesc.matches(desc))
            {
                // Don't reuse the actively debugged resource to avoid stomping
                // it
                if (_markedDebugName.has_value() &&
                    _aliasedDebugNames[i].ends_with(*_markedDebugName))
                {
                    // Make sure we're not just partially matching the last part
                    // of the concatenated debug identifier
                    const size_t breakPosition = _aliasedDebugNames[i].size() -
                                                 1 - _markedDebugName->size();
                    if (_aliasedDebugNames[i].size() ==
                            _markedDebugName->size() ||
                        _aliasedDebugNames[i][breakPosition] == '|')
                        continue;
                }

                _generations[i] &= ~sNotInUseGenerationFlag;
                _framesSinceUsed[i] = 0;

                const Handle handle{
                    .index = i,
                    .generation = _generations[i],
                };

                appendDebugName(handle, debugName);

                return handle;
            }
        }
    }

    uint32_t index = 0xFFFFFFFF;
    if (!_freelist.empty())
        index = _freelist.pop_back();
    else
    {
        _resources.emplace_back();
        _descriptions.emplace_back();
        _aliasedDebugNames.emplace_back(gAllocators.general);
        _debugNames.emplace_back(gAllocators.general);
        _preserved.push_back(false);
        _framesSinceUsed.push_back((uint8_t)0);
        // We might have handle generations from previously destroyed resources
        if (_generations.size() < _resources.size())
        {
            _generations.push_back((uint64_t)sNotInUseGenerationFlag);
        }
        index = asserted_cast<uint32_t>(_resources.size() - 1);
    }
    WHEELS_ASSERT(!resourceInUse(index));
    WHEELS_ASSERT(_resources[index].handle == CppNativeType{});

    _resources[index] = _device->create(CreateInfo{
        .desc = desc,
        .debugName = debugName,
    });
    _descriptions[index] = desc;
    _aliasedDebugNames[index].extend(debugName);
    uint64_t &generation = _generations[index];
    generation = generation & ~sNotInUseGenerationFlag;

    _preserved[index] = false;
    _framesSinceUsed[index] = 0;

    const Handle handle{
        .index = index,
        .generation = _generations[index],
    };

    assertValidHandle(handle);

    appendDebugName(handle, debugName);

    if (_markedDebugName.has_value() && debugName == *_markedDebugName)
        _markedDebugHandle = handle;

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
    WHEELS_ASSERT(_device != nullptr);

    // NOTE:
    // Any changes need to be mirrored in assertValidHandle().
    if (!handle.isValid())
        return false;
    if (handle.index >= _resources.size())
        return false;
    if (handle.index >= _generations.size())
        return false;
    if (_markedDebugHandle.has_value() &&
        handle.index == _markedDebugHandle->index)
    {
        const uint64_t storedGeneration =
            _generations[handle.index] & ~sNotInUseGenerationFlag;
        if (handle.generation != storedGeneration &&
            (handle.generation + 1) != storedGeneration)
            return false;
    }
    else
        // Handle generation matching means held generation isn't flagged unused
        if (handle.generation != _generations[handle.index])
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
    WHEELS_ASSERT(_device != nullptr);
    assertValidHandle(handle);

    return _resources[handle.index].handle;
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
const Resource &RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::resource(Handle handle) const
{
    WHEELS_ASSERT(_device != nullptr);
    assertValidHandle(handle);

    return _resources[handle.index];
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
    WHEELS_ASSERT(_device != nullptr);
    assertValidHandle(handle);

    _resources[handle.index].transition(cb, state);
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
    WHEELS_ASSERT(_device != nullptr);
    assertValidHandle(handle);

    return _resources[handle.index].transitionBarrier(state, force_barrier);
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
    WHEELS_ASSERT(_device != nullptr);
    assertValidHandle(handle);

    wheels::String &aliasedName = _aliasedDebugNames[handle.index];
    if (!aliasedName.empty())
        aliasedName.push_back('|');
    aliasedName.extend(debugName);

    // TODO: Set these at once? Need to be careful to set before
    // submits?
    _device->logical().setDebugUtilsObjectNameEXT(
        vk::DebugUtilsObjectNameInfoEXT{
            .objectType = ObjectType,
            .objectHandle = reinterpret_cast<uint64_t>(
                static_cast<NativeType>(_resources[handle.index].handle)),
            .pObjectName = _aliasedDebugNames[handle.index].c_str(),
        });

    assertUniqueDebugName(debugName);
    _debugNames.emplace_back(gAllocators.general, debugName);

    if (_markedDebugName.has_value() && debugName == *_markedDebugName)
        _markedDebugHandle = handle;
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
void RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::release(Handle handle)
{
    WHEELS_ASSERT(_device != nullptr);
    assertValidHandle(handle);

    // Releases on preserved resources are valid as no-ops so that the info
    // about preserving doesn't have to permeate the renderer.
    if (_preserved[handle.index])
        return;

    _generations[handle.index]++;
    _generations[handle.index] |= sNotInUseGenerationFlag;
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
void RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::preserve(Handle handle)
{
    WHEELS_ASSERT(_device != nullptr);
    assertValidHandle(handle);
    WHEELS_ASSERT(
        !_preserved[handle.index] &&
        "Resource is being preseved in two places, ownership gets muddy.");

    _preserved[handle.index] = true;
    _framesSinceUsed[handle.index] = 0;
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
wheels::Span<const wheels::String> RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::debugNames() const
{
    WHEELS_ASSERT(_device != nullptr);

    return _debugNames;
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
Handle RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::activeDebugHandle() const
{
    WHEELS_ASSERT(_device != nullptr);

    if (!_markedDebugHandle.has_value() || !isValidHandle(*_markedDebugHandle))
        return Handle{};

    return *_markedDebugHandle;
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
wheels::Optional<wheels::StrSpan> RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::activeDebugName() const
{
    WHEELS_ASSERT(_device != nullptr);

    if (_markedDebugName.has_value())
        return wheels::Optional<wheels::StrSpan>{*_markedDebugName};

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
    WHEELS_ASSERT(_device != nullptr);

    _markedDebugName = wheels::String{gAllocators.general, debugName};
    // Let's not worry about finding the resource immediately, we'll have it on
    // the next frame.
    _markedDebugHandle.reset();
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
void RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::clearDebug()
{
    WHEELS_ASSERT(_device != nullptr);

    _markedDebugName.reset();
    _markedDebugHandle.reset();
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
    WHEELS_ASSERT(handle.index < _resources.size());
    WHEELS_ASSERT(handle.index < _generations.size());
    if (_markedDebugHandle.has_value() &&
        handle.index == _markedDebugHandle->index)
    {
#ifndef NDEBUG
        const uint64_t storedGeneration =
            _generations[handle.index] & ~sNotInUseGenerationFlag;
        WHEELS_ASSERT(
            handle.generation == storedGeneration ||
            (handle.generation + 1) == storedGeneration);
#endif // NDEBUG
    }
    else
        // Handle generation matching means held generation isn't flagged unused
        WHEELS_ASSERT(handle.generation == _generations[handle.index]);
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
bool RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::resourceInUse(uint32_t i) const
{
    WHEELS_ASSERT(i < _generations.size());
    return (_generations[i] & sNotInUseGenerationFlag) == 0;
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
    for (const wheels::String &name : _debugNames)
        WHEELS_ASSERT(
            debugName != name &&
            "Debug names need to be unique within a frame");
#else
    (void)debugName;
#endif // NDEBUG
}

#endif // PROSPER_RENDER_RESOURCE_COLLECTION_HPP
