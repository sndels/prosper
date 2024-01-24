#ifndef PROSPER_RENDER_MESHLET_CULLER_HPP
#define PROSPER_RENDER_MESHLET_CULLER_HPP

#include "../gfx/Fwd.hpp"
#include "../gfx/Resources.hpp"
#include "../scene/Fwd.hpp"
#include "../utils/Fwd.hpp"
#include "../utils/Utils.hpp"
#include "Fwd.hpp"
#include "RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/inline_array.hpp>
#include <wheels/containers/static_array.hpp>

struct MeshletCullerOutput
{
    BufferHandle dataBuffer;
    BufferHandle argumentBuffer;
};

class MeshletCuller
{
  public:
    MeshletCuller(Device *device, RenderResources *resources);
    ~MeshletCuller();

    MeshletCuller(const MeshletCuller &other) = delete;
    MeshletCuller(MeshletCuller &&other) = delete;
    MeshletCuller &operator=(const MeshletCuller &other) = delete;
    MeshletCuller &operator=(MeshletCuller &&other) = delete;

    void startFrame();

    enum class Mode
    {
        Opaque,
        Transparent,
    };
    [[nodiscard]] MeshletCullerOutput record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb, Mode mode,
        const World &world, const Camera &cam, uint32_t nextFrame,
        const char *debugPrefix, SceneStats *sceneStats, Profiler *profiler);

  private:
    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    // Keep this a tight upper bound or make arrays dynamic if usage varies a
    // lot based on content
    static const uint32_t sMaxRecordsPerFrame = 2;
    uint32_t _currentFrameRecordCount{0};
    wheels::StaticArray<
        wheels::InlineArray<Buffer, sMaxRecordsPerFrame>, MAX_FRAMES_IN_FLIGHT>
        _dataBuffers;
    wheels::StaticArray<
        wheels::InlineArray<Buffer, sMaxRecordsPerFrame>, MAX_FRAMES_IN_FLIGHT>
        _argumentBuffers;
};

#endif // PROSPER_RENDER_MESHLET_CULLER_HPP
