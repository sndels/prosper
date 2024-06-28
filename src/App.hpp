#ifndef PROSPER_APP_HPP
#define PROSPER_APP_HPP

#include "Window.hpp"
#include "gfx/Device.hpp"
#include "gfx/RingBuffer.hpp"
#include "gfx/Swapchain.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "scene/Camera.hpp"
#include "scene/DrawType.hpp"
#include "scene/Fwd.hpp"
#include "utils/Profiler.hpp"
#include "utils/SceneStats.hpp"
#include "utils/Timer.hpp"

#include <filesystem>
#include <future>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/allocators/tlsf_allocator.hpp>
#include <wheels/containers/static_array.hpp>
#include <wheels/owning_ptr.hpp>

class App
{
  public:
    struct Settings
    {
        std::filesystem::path scene;
        Device::Settings device;
    };

    App(std::filesystem::path scenePath) noexcept;
    ~App();

    App(const App &other) = delete;
    App(App &&other) = delete;
    App &operator=(const App &other) = delete;
    App &operator=(App &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);
    void setInitScratchHighWatermark(size_t value);
    void run();

  private:
    void recompileShaders(wheels::ScopedScratch scopeALloc);
    void recreateSwapchainAndRelated(wheels::ScopedScratch scopeAlloc);
    void recreateViewportRelated();

    void handleMouseGestures();
    void handleKeyboardInput(float deltaS);

    void drawFrame(
        wheels::ScopedScratch scopeAlloc, uint32_t scopeHighWatermark);

    uint32_t nextSwapchainImage(
        wheels::ScopedScratch scopeAlloc, uint32_t nextFrame);

    float currentTimelineTimeS() const;

    void capFramerate();

    struct UiChanges
    {
        bool rtDirty{false};
        bool timeTweaked{false};
    };
    UiChanges drawUi(
        wheels::ScopedScratch scopeAlloc, uint32_t nextFrame,
        const wheels::Array<Profiler::ScopeData> &profilerDatas,
        uint32_t scopeHighWatermark);
    void drawOptions();
    void drawRendererSettings(UiChanges &uiChanges);
    void drawProfiling(
        wheels::ScopedScratch scopeAlloc,
        const wheels::Array<Profiler::ScopeData> &profilerDatas);
    void drawMemory(uint32_t scopeHighWatermark) const;
    // Returns true if time was tweaked
    bool drawTimeline();
    // Returns true if settings changed
    bool drawCameraUi();
    void drawSceneStats(uint32_t nextFrame) const;

    void updateDebugLines(const Scene &scene, uint32_t nextFrame);

    struct RenderIndices
    {
        uint32_t nextFrame{0xFFFF'FFFF};
        uint32_t nextImage{0xFFFF'FFFF};
    };
    void render(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const vk::Rect2D &renderArea, const RenderIndices &indices,
        const UiChanges &uiChanges);
    [[nodiscard]] ImageHandle blitColorToFinalComposite(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        ImageHandle toneMapped);
    void blitFinalComposite(
        vk::CommandBuffer cb, ImageHandle finalComposite, uint32_t nextImage);
    void readbackDrawStats(
        vk::CommandBuffer cb, uint32_t nextFrame, BufferHandle srcBuffer);
    // Returns true if present succeeded, false if swapchain should be recreated
    [[nodiscard]] bool submitAndPresent(
        vk::CommandBuffer cb, uint32_t nextFrame);
    void handleResizes(
        wheels::ScopedScratch scopeAlloc, bool shouldResizeSwapchain);

    // Separate allocator for async polling as TlsfAllocator is not thread safe
    wheels::TlsfAllocator m_fileChangePollingAlloc;
    std::filesystem::path m_scenePath;

    // This allocator should only be used for the descriptors that can live
    // until the end of the program. As such, reset() shouldn't be called so
    // that users can rely on the descriptors being there once allocated.
    wheels::OwningPtr<DescriptorAllocator> m_staticDescriptorsAlloc;

    wheels::OwningPtr<Swapchain> m_swapchain;
    wheels::StaticArray<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT>
        m_commandBuffers;

    vk::Extent2D m_viewportExtent{};

    // TODO:
    // Should this be a global too?
    RingBuffer m_constantsRing;

    wheels::OwningPtr<Camera> m_cam;
    wheels::OwningPtr<World> m_world;

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
    wheels::OwningPtr<MeshletCuller> m_meshletCuller;
    wheels::OwningPtr<TextureReadback> m_textureReadback;

    bool m_useFpsLimit{true};
    int32_t m_fpsLimit{140};
    bool m_recompileShaders{false};
    bool m_referenceRt{false};
    bool m_renderDeferred{true};
    bool m_deferredRt{false};
    bool m_renderDoF{false};
    bool m_textureDebugActive{false};
    bool m_drawUi{true};
    bool m_forceViewportRecreate{false};
    bool m_forceCamUpdate{true};
    bool m_applyIbl{false};
    bool m_sceneChanged{false};
    bool m_newSceneDataLoaded{false};
    bool m_applyTaa{true};
    bool m_applyJitter{true};
    DrawType m_drawType{DrawType::Default};

    bool m_camFreeLook{false};
    CameraTransform m_sceneCameraTransform;
    CameraParameters m_cameraParameters;
    wheels::Optional<FrustumCorners> m_debugFrustum;
    glm::vec3 m_frustumDebugColor{1.f, 1.f, 1.f};
    bool m_pickFocusDistance{false};
    glm::vec2 m_pickedFocusPx{-1.f, -1.f};
    bool m_waitFocusDistance{false};

    wheels::StaticArray<SceneStats, MAX_FRAMES_IN_FLIGHT> m_sceneStats;
    wheels::StaticArray<BufferHandle, MAX_FRAMES_IN_FLIGHT> m_drawStats;

    std::chrono::high_resolution_clock::time_point m_lastTimeChange;
    float m_timeOffsetS{0.f};
    bool m_isPlaying{false};

    uint32_t m_ctorScratchHighWatermark{0};

    Timer m_frameTimer;
    std::chrono::time_point<std::chrono::file_clock> m_recompileTime;

    wheels::StaticArray<vk::Semaphore, MAX_FRAMES_IN_FLIGHT>
        m_imageAvailableSemaphores;
    wheels::StaticArray<vk::Semaphore, MAX_FRAMES_IN_FLIGHT>
        m_renderFinishedSemaphores;

    std::future<wheels::HashSet<std::filesystem::path>> m_fileChanges;
};

#endif // PROSPER_APP_HPP
