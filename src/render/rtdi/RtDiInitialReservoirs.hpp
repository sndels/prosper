#ifndef PROSPER_RENDER_RTDI_INITIAL_RESERVOIRS_HPP
#define PROSPER_RENDER_RTDI_INITIAL_RESERVOIRS_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

#include "../../gfx/Device.hpp"
#include "../../scene/Camera.hpp"
#include "../../scene/World.hpp"
#include "../../utils/Profiler.hpp"
#include "../ComputePass.hpp"
#include "../GBufferRenderer.hpp"
#include "../RenderResources.hpp"

class RtDiInitialReservoirs
{
  public:
    struct InputDSLayouts
    {
        vk::DescriptorSetLayout camera;
        const World::DSLayouts &world;
    };
    RtDiInitialReservoirs(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc,
        const InputDSLayouts &dsLayouts);

    ~RtDiInitialReservoirs() = default;

    RtDiInitialReservoirs(const RtDiInitialReservoirs &other) = delete;
    RtDiInitialReservoirs(RtDiInitialReservoirs &&other) = delete;
    RtDiInitialReservoirs &operator=(const RtDiInitialReservoirs &other) =
        delete;
    RtDiInitialReservoirs &operator=(RtDiInitialReservoirs &&other) = delete;

    // Returns true if recompile happened
    bool recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        const InputDSLayouts &dsLayouts);

    struct Output
    {
        ImageHandle reservoirs;
    };
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, const World &world, const Camera &cam,
        const GBufferRenderer::Output &gbuffer, uint32_t nextFrame,
        Profiler *profiler);

    RenderResources *_resources{nullptr};
    ComputePass _computePass;

    uint32_t _frameIndex{0};
};

#endif // PROSPER_RENDER_RT_DI_INITIAL_RESERVOIRS_HPP
