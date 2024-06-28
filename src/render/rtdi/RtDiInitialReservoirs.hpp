#ifndef PROSPER_RENDER_RTDI_INITIAL_RESERVOIRS_HPP
#define PROSPER_RENDER_RTDI_INITIAL_RESERVOIRS_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

#include "../../gfx/Fwd.hpp"
#include "../../scene/Fwd.hpp"
#include "../../utils/Fwd.hpp"
#include "../ComputePass.hpp"
#include "../Fwd.hpp"
#include "../RenderResourceHandle.hpp"

class RtDiInitialReservoirs
{
  public:
    RtDiInitialReservoirs() noexcept = default;
    ~RtDiInitialReservoirs() = default;

    RtDiInitialReservoirs(const RtDiInitialReservoirs &other) = delete;
    RtDiInitialReservoirs(RtDiInitialReservoirs &&other) = delete;
    RtDiInitialReservoirs &operator=(const RtDiInitialReservoirs &other) =
        delete;
    RtDiInitialReservoirs &operator=(RtDiInitialReservoirs &&other) = delete;

    struct InputDSLayouts
    {
        vk::DescriptorSetLayout camera;
        const WorldDSLayouts &world;
    };
    void init(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc,
        const InputDSLayouts &dsLayouts);

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
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const World &world, const Camera &cam,
        const GBufferRendererOutput &gbuffer, uint32_t nextFrame);

    bool m_initialized{false};
    ComputePass m_computePass;

    uint32_t m_frameIndex{0};
};

#endif // PROSPER_RENDER_RT_DI_INITIAL_RESERVOIRS_HPP
