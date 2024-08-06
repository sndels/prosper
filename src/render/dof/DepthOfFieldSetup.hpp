#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_SETUP_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_SETUP_HPP

#include "../../gfx/Fwd.hpp"
#include "../../scene/Fwd.hpp"
#include "../../utils/Fwd.hpp"
#include "../../utils/Utils.hpp"
#include "../ComputePass.hpp"
#include "../Fwd.hpp"
#include "../RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

class DepthOfFieldSetup
{
  public:
    DepthOfFieldSetup() noexcept = default;
    ~DepthOfFieldSetup() = default;

    DepthOfFieldSetup(const DepthOfFieldSetup &other) = delete;
    DepthOfFieldSetup(DepthOfFieldSetup &&other) = delete;
    DepthOfFieldSetup &operator=(const DepthOfFieldSetup &other) = delete;
    DepthOfFieldSetup &operator=(DepthOfFieldSetup &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDsLayout);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDsLayout);

    struct Input
    {
        ImageHandle illumination;
        ImageHandle depth;
    };
    struct Output
    {
        ImageHandle halfResIllumination;
        ImageHandle halfResCircleOfConfusion;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const Camera &cam, const Input &input, uint32_t nextFrame);

  private:
    bool m_initialized{false};
    ComputePass m_computePass;
};

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_SETUP_HPP
