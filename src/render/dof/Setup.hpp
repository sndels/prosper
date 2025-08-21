#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_SETUP_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_SETUP_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "scene/Fwd.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

namespace render::dof
{

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

class Setup
{
  public:
    Setup() noexcept = default;
    ~Setup() = default;

    Setup(const Setup &other) = delete;
    Setup(Setup &&other) = delete;
    Setup &operator=(const Setup &other) = delete;
    Setup &operator=(Setup &&other) = delete;

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
        const scene::Camera &cam, const Input &input, uint32_t nextFrame);

  private:
    bool m_initialized{false};
    ComputePass m_computePass;
};

} // namespace render::dof

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_SETUP_HPP
