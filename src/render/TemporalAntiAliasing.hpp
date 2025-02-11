#ifndef PROSPER_RENDER_TEMPORAL_ANTI_ALIASING_HPP
#define PROSPER_RENDER_TEMPORAL_ANTI_ALIASING_HPP

#include "RenderResourceHandle.hpp"
#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "scene/Fwd.hpp"
#include "utils/ForEach.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

#define COLOR_CLIPPING_TYPES None, MinMax, Variance
#define COLOR_CLIPPING_TYPES_AND_COUNT COLOR_CLIPPING_TYPES, Count
#define COLOR_CLIPPING_TYPES_STRINGIFY(t) #t,
#define COLOR_CLIPPING_TYPE_STRS                                               \
    FOR_EACH(COLOR_CLIPPING_TYPES_STRINGIFY, COLOR_CLIPPING_TYPES)

#define VELOCITY_SAMPLING_TYPES Center, Largest, Closest
#define VELOCITY_SAMPLING_TYPES_AND_COUNT VELOCITY_SAMPLING_TYPES, Count
#define VELOCITY_SAMPLING_TYPES_STRINGIFY(t) #t,
#define VELOCITY_SAMPLING_TYPE_STRS                                            \
    FOR_EACH(VELOCITY_SAMPLING_TYPES_STRINGIFY, VELOCITY_SAMPLING_TYPES)

class TemporalAntiAliasing
{
  public:
    // NOLINTNEXTLINE(performance-enum-size) specialization constant
    enum class ColorClippingType : uint32_t
    {
        COLOR_CLIPPING_TYPES_AND_COUNT
    };

    // NOLINTNEXTLINE(performance-enum-size) specialization constant
    enum class VelocitySamplingType : uint32_t
    {
        VELOCITY_SAMPLING_TYPES_AND_COUNT
    };

    TemporalAntiAliasing() noexcept = default;
    ~TemporalAntiAliasing() = default;

    TemporalAntiAliasing(const TemporalAntiAliasing &other) = delete;
    TemporalAntiAliasing(TemporalAntiAliasing &&other) = delete;
    TemporalAntiAliasing &operator=(const TemporalAntiAliasing &other) = delete;
    TemporalAntiAliasing &operator=(TemporalAntiAliasing &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDsLayout);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDSLayout);

    void drawUi();

    struct Input
    {
        ImageHandle illumination;
        ImageHandle velocity;
        ImageHandle depth;
    };
    struct Output
    {
        ImageHandle resolvedIllumination;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const Camera &cam, const Input &input, uint32_t nextFrame);
    void releasePreserved();

  private:
    bool m_initialized{false};
    ComputePass m_computePass;

    ImageHandle m_previousResolveOutput;
    ColorClippingType m_colorClipping{ColorClippingType::Variance};
    VelocitySamplingType m_velocitySampling{VelocitySamplingType::Closest};
    bool m_catmullRom{true};
    bool m_luminanceWeighting{true};
};

#endif // PROSPER_RENDER_TEMPORAL_ANTI_ALIASING_HPP
