#ifndef PROSPER_RENDER_IMAGE_BASED_LIGHTING_HPP
#define PROSPER_RENDER_IMAGE_BASED_LIGHTING_HPP

#include "gfx/Fwd.hpp"
#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "scene/Fwd.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

class ImageBasedLighting
{
  public:
    ImageBasedLighting() noexcept = default;
    ~ImageBasedLighting() = default;

    ImageBasedLighting(const ImageBasedLighting &other) = delete;
    ImageBasedLighting(ImageBasedLighting &&other) = delete;
    ImageBasedLighting &operator=(const ImageBasedLighting &other) = delete;
    ImageBasedLighting &operator=(ImageBasedLighting &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    [[nodiscard]] bool isGenerated() const;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    void recordGeneration(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb, World &world,
        uint32_t nextFrame);

  private:
    bool m_initialized{false};
    ComputePass m_sampleIrradiance;
    ComputePass m_integrateSpecularBrdf;
    ComputePass m_prefilterRadiance;

    bool m_generated{false};
};

#endif // PROSPER_RENDER_IMAGE_BASED_LIGHTING_HPP
