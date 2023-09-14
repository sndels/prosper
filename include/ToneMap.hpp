#ifndef PROSPER_TONE_MAP_HPP
#define PROSPER_TONE_MAP_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

#include "ComputePass.hpp"
#include "Device.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "Utils.hpp"

class ToneMap
{
  public:
    ToneMap(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources,
        DescriptorAllocator *staticDescriptorsAlloc);
    ~ToneMap() = default;

    ToneMap(const ToneMap &other) = delete;
    ToneMap(ToneMap &&other) = delete;
    ToneMap &operator=(const ToneMap &other) = delete;
    ToneMap &operator=(ToneMap &&other) = delete;

    void recompileShaders(wheels::ScopedScratch scopeAlloc);

    void drawUi();

    struct Output
    {
        ImageHandle toneMapped;
    };
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, ImageHandle inColor, uint32_t nextFrame,
        Profiler *profiler);

  private:
    Output createOutputs(const vk::Extent2D &size);

    RenderResources *_resources{nullptr};
    ComputePass _computePass;

    float _exposure{1.f};
};

#endif // PROSPER_TONE_MAP_HPP
