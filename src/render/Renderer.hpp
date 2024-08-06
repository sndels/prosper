#ifndef PROSPER_RENDER_RENDERER_HPP
#define PROSPER_RENDER_RENDERER_HPP

#include "gfx/Fwd.hpp"
#include "render/DrawStats.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "scene/DrawType.hpp"
#include "scene/Fwd.hpp"
#include "utils/Utils.hpp"

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/hash_set.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>
#include <wheels/owning_ptr.hpp>

class Renderer
{
  public:
    Renderer() noexcept;
    ~Renderer();

    Renderer(const Renderer &other) = delete;
    Renderer(Renderer &&other) = delete;
    Renderer &operator=(const Renderer &other) = delete;
    Renderer &operator=(Renderer &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc,
        const SwapchainConfig &swapchainConfig,
        vk::DescriptorSetLayout camDsLayout,
        const WorldDSLayouts &worldDsLayouts);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDsLayout,
        const WorldDSLayouts &worldDsLayouts,
        const wheels::HashSet<std::filesystem::path> &changedFiles);
    static void recreateSwapchainAndRelated();
    void recreateViewportRelated();

    void startFrame();
    // Returns true if rt should be marked dirty
    [[nodiscard]] bool drawUi(Camera &cam);

    [[nodiscard]] const DrawStats &drawStats(uint32_t nextFrame);
    [[nodiscard]] const vk::Extent2D &viewportExtentInUi() const;
    // Returns true if the held viewport extent doesn't match the current one
    [[nodiscard]] bool viewportResized() const;
    [[nodiscard]] glm::vec2 viewportOffsetInUi() const;
    [[nodiscard]] float lodBias() const;
    [[nodiscard]] bool rtInUse() const;
    [[nodiscard]] wheels::Optional<glm::vec4> tryDepthReadback();
    [[nodiscard]] bool depthAvailable() const;

    struct Options
    {
        bool rtDirty{false};
        bool drawUi{false};
        wheels::Optional<glm::vec2> readbackDepthPx;
    };
    void render(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const Camera &cam, World &world, const vk::Rect2D &renderArea,
        const SwapchainImage &swapImage, uint32_t nextFrame,
        const Options &options);

  private:
    [[nodiscard]] ImageHandle blitColorToFinalComposite(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        ImageHandle toneMapped, const vk::Extent2D &swapImageExtent,
        bool drawUi);
    void readbackDrawStats(
        vk::CommandBuffer cb, uint32_t nextFrame, BufferHandle srcBuffer);

    wheels::OwningPtr<MeshletCuller> m_meshletCuller;
    wheels::OwningPtr<HierarchicalDepthDownsampler>
        m_hierarchicalDepthDownsampler;
    wheels::OwningPtr<LightClustering> m_lightClustering;
    wheels::OwningPtr<ForwardRenderer> m_forwardRenderer;
    wheels::OwningPtr<GBufferRenderer> m_gbufferRenderer;
    wheels::OwningPtr<DeferredShading> m_deferredShading;
    wheels::OwningPtr<RtDirectIllumination> m_rtDirectIllumination;
    wheels::OwningPtr<RtReference> m_rtReference;
    wheels::OwningPtr<SkyboxRenderer> m_skyboxRenderer;
    wheels::OwningPtr<DebugRenderer> m_debugRenderer;
    wheels::OwningPtr<ToneMap> m_toneMap;
    wheels::OwningPtr<ImGuiRenderer> m_imguiRenderer;
    wheels::OwningPtr<TextureDebug> m_textureDebug;
    wheels::OwningPtr<DepthOfField> m_depthOfField;
    wheels::OwningPtr<ImageBasedLighting> m_imageBasedLighting;
    wheels::OwningPtr<TemporalAntiAliasing> m_temporalAntiAliasing;
    wheels::OwningPtr<TextureReadback> m_textureReadback;

    wheels::StaticArray<DrawStats, MAX_FRAMES_IN_FLIGHT> m_drawStats;
    wheels::StaticArray<BufferHandle, MAX_FRAMES_IN_FLIGHT> m_gpuDrawStats;

    vk::Extent2D m_viewportExtentInUi{};

    bool m_textureDebugActive{false};
    bool m_referenceRt{false};
    bool m_renderDeferred{true};
    bool m_deferredRt{false};
    bool m_renderDoF{false};
    bool m_applyIbl{false};
    bool m_applyTaa{true};
    bool m_applyJitter{true};
    DrawType m_drawType{DrawType::Default};
};

#endif // PROSPER_RENDER_RENDERER_HPP
