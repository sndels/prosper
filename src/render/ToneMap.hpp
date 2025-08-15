#ifndef PROSPER_RENDER_TONE_MAP_HPP
#define PROSPER_RENDER_TONE_MAP_HPP

#include "RenderResourceHandle.hpp"
#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "scene/Texture.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

namespace render
{

class ToneMap
{
  public:
    ToneMap() noexcept = default;
    ~ToneMap() = default;

    ToneMap(const ToneMap &other) = delete;
    ToneMap(ToneMap &&other) = delete;
    ToneMap &operator=(const ToneMap &other) = delete;
    ToneMap &operator=(ToneMap &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

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
        ImageHandle inColor, uint32_t nextFrame);

  private:
    bool m_initialized{false};
    ComputePass m_computePass;
    scene::Texture3D m_lut;

    float m_exposure{1.f};
    float m_contrast{1.f};
};

} // namespace render

#endif // PROSPER_RENDER_TONE_MAP_HPP
