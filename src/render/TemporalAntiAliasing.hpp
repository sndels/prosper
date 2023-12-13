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

#include "../utils/ForEach.hpp"

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
    enum class ColorClippingType : uint32_t
    {
        COLOR_CLIPPING_TYPES_AND_COUNT
    };

    enum class VelocitySamplingType : uint32_t
    {
        VELOCITY_SAMPLING_TYPES_AND_COUNT
    };

    TemporalAntiAliasing(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc,
        vk::DescriptorSetLayout camDsLayout);
    ~TemporalAntiAliasing() = default;

    TemporalAntiAliasing(const TemporalAntiAliasing &other) = delete;
    TemporalAntiAliasing(TemporalAntiAliasing &&other) = delete;
    TemporalAntiAliasing &operator=(const TemporalAntiAliasing &other) = delete;
    TemporalAntiAliasing &operator=(TemporalAntiAliasing &&other) = delete;

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
        const Camera &cam, const Input &input, uint32_t nextFrame,
        Profiler *profiler);
    void releasePreserved();

  private:
    Output createOutputs(const vk::Extent2D &size);

    RenderResources *_resources{nullptr};
    ComputePass _computePass;

    ImageHandle _previousResolveOutput;
    ColorClippingType _colorClipping{ColorClippingType::Variance};
    VelocitySamplingType _velocitySampling{VelocitySamplingType::Closest};
    bool _catmullRom{true};
};

#endif // PROSPER_RENDER_TEMPORAL_ANTI_ALIASING_HPP
