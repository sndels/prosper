#ifndef PROSPER_RENDER_TONE_MAP_HPP
#define PROSPER_RENDER_TONE_MAP_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

#include "../gfx/Fwd.hpp"
#include "../scene/Texture.hpp"
#include "../utils/Fwd.hpp"
#include "../utils/Utils.hpp"
#include "ComputePass.hpp"
#include "Fwd.hpp"
#include "RenderResourceHandle.hpp"

class ToneMap
{
  public:
    ToneMap() noexcept = default;
    ~ToneMap() = default;

    ToneMap(const ToneMap &other) = delete;
    ToneMap(ToneMap &&other) = delete;
    ToneMap &operator=(const ToneMap &other) = delete;
    ToneMap &operator=(ToneMap &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources,
        DescriptorAllocator *staticDescriptorsAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    void drawUi();

    struct Output
    {
        ImageHandle toneMapped;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        ImageHandle inColor, uint32_t nextFrame, Profiler *profiler);

  private:
    Output createOutputs(const vk::Extent2D &size);

    bool _initialized{false};
    RenderResources *_resources{nullptr};
    ComputePass _computePass;
    Texture3D _lut;

    float _exposure{1.f};
    float _contrast{1.f};
};

#endif // PROSPER_RENDER_TONE_MAP_HPP
