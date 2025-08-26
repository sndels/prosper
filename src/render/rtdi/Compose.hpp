#ifndef PROSPER_RENDER_RTDI_COMPOSE_HPP
#define PROSPER_RENDER_RTDI_COMPOSE_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "render/rtdi/Trace.hpp"

#include <wheels/allocators/scoped_scratch.hpp>

namespace render::rtdi
{

class Compose
{
  public:
    Compose() noexcept = default;
    ~Compose() = default;

    Compose(const Compose &other) = delete;
    Compose(Compose &&other) = delete;
    Compose &operator=(const Compose &other) = delete;
    Compose &operator=(Compose &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    // Returns true if recompile happened
    bool recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    using Input = Trace::Output;
    struct Output
    {
        ImageHandle illumination;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const Input &input, uint32_t nextFrame);

    bool m_initialized{false};
    ComputePass m_computePass;
};

} // namespace render::rtdi

#endif // PROSPER_RENDER_RT_DI_COMPOSE_HPP
