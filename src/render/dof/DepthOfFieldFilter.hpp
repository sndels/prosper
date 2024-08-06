#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_FILTER_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_FILTER_HPP

#include "gfx/Fwd.hpp"
#include "gfx/Resources.hpp"
#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "utils/Fwd.hpp"

#include <wheels/allocators/scoped_scratch.hpp>

class DepthOfFieldFilter
{
  public:
    DepthOfFieldFilter() noexcept = default;
    ~DepthOfFieldFilter() = default;

    DepthOfFieldFilter(const DepthOfFieldFilter &other) = delete;
    DepthOfFieldFilter(DepthOfFieldFilter &&other) = delete;
    DepthOfFieldFilter &operator=(const DepthOfFieldFilter &other) = delete;
    DepthOfFieldFilter &operator=(DepthOfFieldFilter &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    void startFrame();

    struct Output
    {
        ImageHandle filteredIlluminationWeight;
    };
    struct DebugNames
    {
        const char *scope{nullptr};
        const char *outRes{nullptr};
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        ImageHandle inIlluminationWeight, uint32_t nextFrame,
        const DebugNames &debugNames);

  private:
    bool m_initialized{false};
    ComputePass m_computePass;
};

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_FILTER_HPP
