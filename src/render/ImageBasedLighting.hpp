#ifndef PROSPER_RENDER_IMAGE_BASED_LIGHTING_HPP
#define PROSPER_RENDER_IMAGE_BASED_LIGHTING_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

#include "../gfx/Fwd.hpp"
#include "../scene/Fwd.hpp"
#include "ComputePass.hpp"
#include "Fwd.hpp"

class ImageBasedLighting
{
  public:
    ImageBasedLighting(
        wheels::ScopedScratch scopeAlloc, Device *device,
        DescriptorAllocator *staticDescriptorsAlloc);
    ~ImageBasedLighting() = default;

    ImageBasedLighting(const ImageBasedLighting &other) = delete;
    ImageBasedLighting(ImageBasedLighting &&other) = delete;
    ImageBasedLighting &operator=(const ImageBasedLighting &other) = delete;
    ImageBasedLighting &operator=(ImageBasedLighting &&other) = delete;

    [[nodiscard]] bool isGenerated() const;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    void recordGeneration(
        vk::CommandBuffer cb, World &world, uint32_t nextFrame,
        Profiler *profiler);

  private:
    void createOutputs();

    Device *_device{nullptr};
    ComputePass _sampleIrradiance;
    ComputePass _integrateSpecularBrdf;
    ComputePass _prefilterRadiance;

    bool _generated{false};
};

#endif // PROSPER_RENDER_IMAGE_BASED_LIGHTING_HPP
