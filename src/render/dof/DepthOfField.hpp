#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

#include "../../gfx/Fwd.hpp"
#include "../../scene/Fwd.hpp"
#include "../../utils/Fwd.hpp"
#include "../../utils/Utils.hpp"
#include "../Fwd.hpp"
#include "../RenderResourceHandle.hpp"
#include "DepthOfFieldCombine.hpp"
#include "DepthOfFieldDilate.hpp"
#include "DepthOfFieldFilter.hpp"
#include "DepthOfFieldFlatten.hpp"
#include "DepthOfFieldGather.hpp"
#include "DepthOfFieldReduce.hpp"
#include "DepthOfFieldSetup.hpp"

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

class DepthOfField
{
  public:
    // Foreground can have an (almost?) infinitely larger bokeh so let's clamp
    // to a smaller but still plausible looking factor
    static constexpr float sMaxFgCoCFactor = 2.f;

    DepthOfField() noexcept = default;
    ~DepthOfField() = default;

    DepthOfField(const DepthOfField &other) = delete;
    DepthOfField(DepthOfField &&other) = delete;
    DepthOfField &operator=(const DepthOfField &other) = delete;
    DepthOfField &operator=(DepthOfField &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc,
        vk::DescriptorSetLayout cameraDsLayout);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout cameraDsLayout);

    void startFrame();

    using Input = DepthOfFieldSetup::Input;
    using Output = DepthOfFieldCombine::Output;
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const Camera &cam, const Input &input, uint32_t nextFrame,
        Profiler *profiler);

  private:
    bool _initialized{false};
    RenderResources *_resources{nullptr};

    DepthOfFieldSetup _setupPass;
    DepthOfFieldReduce _reducePass;
    DepthOfFieldFlatten _flattenPass;
    DepthOfFieldDilate _dilatePass;
    DepthOfFieldGather _gatherPass;
    DepthOfFieldFilter _filterPass;
    DepthOfFieldCombine _combinePass;
};

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_HPP
