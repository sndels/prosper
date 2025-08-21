#ifndef PROSPER_RENDER_RTDI_INITIAL_RESERVOIRS_HPP
#define PROSPER_RENDER_RTDI_INITIAL_RESERVOIRS_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "scene/Fwd.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

namespace render::rtdi
{

class InitialReservoirs
{
  public:
    InitialReservoirs() noexcept = default;
    ~InitialReservoirs() = default;

    InitialReservoirs(const InitialReservoirs &other) = delete;
    InitialReservoirs(InitialReservoirs &&other) = delete;
    InitialReservoirs &operator=(const InitialReservoirs &other) = delete;
    InitialReservoirs &operator=(InitialReservoirs &&other) = delete;

    struct InputDSLayouts
    {
        vk::DescriptorSetLayout camera;
        const scene::WorldDSLayouts &world;
    };
    void init(
        wheels::ScopedScratch scopeAlloc, const InputDSLayouts &dsLayouts);

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
        const scene::World &world, const scene::Camera &cam,
        const GBufferRendererOutput &gbuffer, uint32_t nextFrame);

    bool m_initialized{false};
    ComputePass m_computePass;

    uint32_t m_frameIndex{0};
};

} // namespace render::rtdi

#endif // PROSPER_RENDER_RT_DI_INITIAL_RESERVOIRS_HPP
