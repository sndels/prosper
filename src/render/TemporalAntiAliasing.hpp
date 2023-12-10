#ifndef PROSPER_RENDER_TEMPORAL_ANTI_ALIASING_HPP
#define PROSPER_RENDER_TEMPORAL_ANTI_ALIASING_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

#include "../gfx/Fwd.hpp"
#include "../scene/Fwd.hpp"
#include "../utils/Fwd.hpp"
#include "../utils/Utils.hpp"
#include "ComputePass.hpp"
#include "Fwd.hpp"
#include "RenderResourceHandle.hpp"

class TemporalAntiAliasing
{
  public:
    TemporalAntiAliasing(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources,
        DescriptorAllocator *staticDescriptorsAlloc);
    ~TemporalAntiAliasing() = default;

    TemporalAntiAliasing(const TemporalAntiAliasing &other) = delete;
    TemporalAntiAliasing(TemporalAntiAliasing &&other) = delete;
    TemporalAntiAliasing &operator=(const TemporalAntiAliasing &other) = delete;
    TemporalAntiAliasing &operator=(TemporalAntiAliasing &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDSLayout);

    struct Output
    {
        ImageHandle resolvedIllumination;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        ImageHandle inIllumination, uint32_t nextFrame, Profiler *profiler);
    void releasePreserved();

  private:
    Output createOutputs(const vk::Extent2D &size);

    RenderResources *_resources{nullptr};
    ComputePass _computePass;

    ImageHandle _previousResolveOutput;
};

#endif // PROSPER_RENDER_TEMPORAL_ANTI_ALIASING_HPP
