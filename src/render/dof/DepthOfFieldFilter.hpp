#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_FILTER_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_FILTER_HPP

#include <wheels/allocators/scoped_scratch.hpp>

#include "../../gfx/Fwd.hpp"
#include "../../gfx/Resources.hpp"
#include "../../utils/Fwd.hpp"
#include "../ComputePass.hpp"
#include "../Fwd.hpp"
#include "../RenderResourceHandle.hpp"

class DepthOfFieldFilter
{
  public:
    DepthOfFieldFilter() noexcept = default;
    ~DepthOfFieldFilter() = default;

    DepthOfFieldFilter(const DepthOfFieldFilter &other) = delete;
    DepthOfFieldFilter(DepthOfFieldFilter &&other) = delete;
    DepthOfFieldFilter &operator=(const DepthOfFieldFilter &other) = delete;
    DepthOfFieldFilter &operator=(DepthOfFieldFilter &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc);

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
        const DebugNames &debugNames, Profiler *profiler);

  private:
    bool _initialized{false};
    ComputePass _computePass;
};

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_FILTER_HPP