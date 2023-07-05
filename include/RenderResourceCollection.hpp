#ifndef PROSPER_RENDER_RESOURCE_COLLECTION_HPP
#define PROSPER_RENDER_RESOURCE_COLLECTION_HPP

#include "Device.hpp"
#include "Utils.hpp"

#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/static_array.hpp>

#include <vulkan/vulkan.hpp>

// Generation is designed to get incremented each time a handle is released,
// potentially multiple times per frame. A maximum of e.g. 256 generations
// would enough within a frame but we should also assert against using stale
// handles from previous frames. Hence uint64_t.
// TODO:
// Handle stale handle validation with less space? Have a wrapping generation of
// sufficient size to assume matching gen is actually the same gen?
template <typename Resource> struct RenderResourceHandle
{
  public:
    static const uint32_t sNullIndex = 0xFFFFFFFF;

    // TODO:
    // Protect these? Clang-tidy doesn't like similar value ctor arguments,
    // friending collection messes up the template signature of this handle as
    // well.
    uint32_t index{sNullIndex};
    uint64_t generation{0};

    [[nodiscard]] bool isValid() const { return index != sNullIndex; }
};

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
    RenderResourceCollection(wheels::Allocator &alloc, Device *device);
    virtual ~RenderResourceCollection();

    RenderResourceCollection(RenderResourceCollection &) = delete;
    RenderResourceCollection(RenderResourceCollection &&) = delete;
    RenderResourceCollection &operator=(RenderResourceCollection &) = delete;
    RenderResourceCollection &operator=(RenderResourceCollection &&) = delete;

    void clearDebugNames();
    virtual void destroyResources();

    [[nodiscard]] Handle create(const Description &desc, const char *debugName);
    [[nodiscard]] CppNativeType nativeHandle(Handle handle) const;
    [[nodiscard]] const Resource &resource(Handle handle) const;
    void transition(
        vk::CommandBuffer cb, Handle handle, const ResourceState &state);
    [[nodiscard]] Barrier transitionBarrier(
        Handle handle, const ResourceState &state);
    void release(Handle handle);

  protected:
    Device *_device{nullptr};
    wheels::Allocator &_alloc;

    void assertValidHandle(Handle handle) const;

  private:
    static const uint64_t sNotInUseGenerationFlag = static_cast<size_t>(1)
                                                    << 63;

    [[nodiscard]] bool resourceInUse(uint32_t i) const;

    // RenderImageCollection depends on returned handle indices being
    // contiguous.
    wheels::Array<Resource> _resources;
    wheels::Array<Description> _descriptions;
    wheels::Array<wheels::String> _debugNames;
    wheels::Array<uint64_t> _generations;
};

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::
    RenderResourceCollection(wheels::Allocator &alloc, Device *device)
: _device{device}
, _alloc{alloc}
, _resources{alloc}
, _descriptions{alloc}
, _debugNames{alloc}
, _generations{alloc}
{
    assert(device != nullptr);
}

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
    CppNativeType, NativeType, ObjectType>::clearDebugNames()
{
    for (wheels::String &str : _debugNames)
        str.clear();
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
void RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::destroyResources()
{
    for (const Resource &res : _resources)
        _device->destroy(res);

    _resources.clear();
    _descriptions.clear();
    _debugNames.clear();
    _generations.clear();
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
    const uint32_t descCount = asserted_cast<uint32_t>(_descriptions.size());
#ifndef NDEBUG
    uint32_t matchingCount = 0;
#endif // NDEBUG
    for (uint32_t i = 0; i < descCount; ++i)
    {
        if (!resourceInUse(i))
        {
            const Description &existingDesc = _descriptions[i];
            if (existingDesc.matches(desc))
            {
                _generations[i] &= ~sNotInUseGenerationFlag;
                _debugNames[i].extend(debugName);

                // TODO: Set these at once? Need to be careful to set before
                // submits?
                _device->logical().setDebugUtilsObjectNameEXT(
                    vk::DebugUtilsObjectNameInfoEXT{
                        .objectType = ObjectType,
                        .objectHandle = reinterpret_cast<uint64_t>(
                            static_cast<NativeType>(_resources[i].handle)),
                        .pObjectName = _debugNames[i].c_str(),
                    });

                return Handle{
                    .index = i,
                    .generation = _generations[i],
                };
            }
        }
#ifndef NDEBUG
        else
        {
            if (_descriptions[i].matches(desc))
                matchingCount++;
        }
#endif // NDEBUG
    }

#ifndef NDEBUG
    assert(
        matchingCount < 64 &&
        "Is this resource not being released after being created?");
#endif // NDEBUG

    _resources.push_back(_device->create(CreateInfo{
        .desc = desc,
        .debugName = debugName,
    }));
    _descriptions.push_back(desc);
    _debugNames.emplace_back(_alloc, debugName);
    // TODO:
    // Allow implicit conversions on push_back since literal suffixes don't seem
    // to be portable? Can conversions be supported for literals only?
    _generations.push_back(static_cast<uint64_t>(0));

    const Handle handle{
        .index = asserted_cast<uint32_t>(_resources.size() - 1),
        .generation = 0,
    };

    assertValidHandle(handle);

    return handle;
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
    transition(vk::CommandBuffer cb, Handle handle, const ResourceState &state)
{
    assertValidHandle(handle);

    _resources[handle.index].transition(cb, state);
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
Barrier RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType,
    ObjectType>::transitionBarrier(Handle handle, const ResourceState &state)
{
    assertValidHandle(handle);

    return _resources[handle.index].transitionBarrier(state);
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

    _generations[handle.index]++;
    _generations[handle.index] |= sNotInUseGenerationFlag;
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
    assert(handle.isValid());
    assert(handle.index < _resources.size());
    assert(handle.index < _generations.size());
    assert(handle.generation == _generations[handle.index]);
    assert(
        resourceInUse(handle.index) &&
        "Release called on an already released resource");
}

template <
    typename Handle, typename Resource, typename Description,
    typename CreateInfo, typename ResourceState, typename Barrier,
    typename CppNativeType, typename NativeType, vk::ObjectType ObjectType>
bool RenderResourceCollection<
    Handle, Resource, Description, CreateInfo, ResourceState, Barrier,
    CppNativeType, NativeType, ObjectType>::resourceInUse(uint32_t i) const
{
    assert(i < _generations.size());
    return (_generations[i] & sNotInUseGenerationFlag) == 0;
}

#endif // PROSPER_RENDER_RESOURCE_COLLECTION_HPP
