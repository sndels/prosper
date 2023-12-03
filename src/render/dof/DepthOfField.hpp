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
#include "DepthOfFieldFlatten.hpp"
#include "DepthOfFieldGather.hpp"
#include "DepthOfFieldSetup.hpp"

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

class DepthOfField
{
  public:
    DepthOfField(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc,
        vk::DescriptorSetLayout cameraDsLayout);

    ~DepthOfField() = default;

    DepthOfField(const DepthOfField &other) = delete;
    DepthOfField(DepthOfField &&other) = delete;
    DepthOfField &operator=(const DepthOfField &other) = delete;
    DepthOfField &operator=(DepthOfField &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout cameraDsLayout);

    void drawUi();

    using Input = DepthOfFieldSetup::Input;
    using Output = DepthOfFieldCombine::Output;
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const Camera &cam, const Input &input, uint32_t nextFrame,
        Profiler *profiler);

  private:
    RenderResources *_resources{nullptr};

    DepthOfFieldSetup _setupPass;
    DepthOfFieldFlatten _flattenPass;
    DepthOfFieldDilate _dilatePass;
    DepthOfFieldGather _gatherPass;
    DepthOfFieldCombine _combinePass;
};

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_HPP
