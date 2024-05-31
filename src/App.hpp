#ifndef PROSPER_APP_HPP
#define PROSPER_APP_HPP

#include "Window.hpp"
#include "gfx/Device.hpp"
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

    App(Settings &&settings) noexcept;
    ~App();

    App(const App &other) = delete;
    App(App &&other) = delete;
    App &operator=(const App &other) = delete;
    App &operator=(App &&other) = delete;

    void init();
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
        uint32_t nextFrame{0xFFFFFFFF};
        uint32_t nextImage{0xFFFFFFFF};
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
    wheels::TlsfAllocator _fileChangePollingAlloc;
    std::filesystem::path _scenePath;

    InputHandler _inputHandler;
    wheels::OwningPtr<Window>
        _window; // Needs to be valid before and after everything else
    wheels::OwningPtr<Device>
        _device; // Needs to be valid before and after all other vk resources

    // This allocator should only be used for the descriptors that can live
    // until the end of the program. As such, reset() shouldn't be called so
    // that users can rely on the descriptors being there once allocated.
    wheels::OwningPtr<DescriptorAllocator> _staticDescriptorsAlloc;

    wheels::OwningPtr<Swapchain> _swapchain;
    wheels::StaticArray<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT>
        _commandBuffers;

    vk::Extent2D _viewportExtent{};

    // Stored here, managed by (earliest) passes that write to them
    wheels::OwningPtr<RenderResources> _resources;

    wheels::OwningPtr<Camera> _cam;
    wheels::OwningPtr<World> _world;

    wheels::OwningPtr<LightClustering> _lightClustering;
    wheels::OwningPtr<ForwardRenderer> _forwardRenderer;
    wheels::OwningPtr<GBufferRenderer> _gbufferRenderer;
    wheels::OwningPtr<DeferredShading> _deferredShading;
    wheels::OwningPtr<RtDirectIllumination> _rtDirectIllumination;
    wheels::OwningPtr<RtReference> _rtReference;
    wheels::OwningPtr<SkyboxRenderer> _skyboxRenderer;
    wheels::OwningPtr<DebugRenderer> _debugRenderer;
    wheels::OwningPtr<ToneMap> _toneMap;
    wheels::OwningPtr<ImGuiRenderer> _imguiRenderer;
    wheels::OwningPtr<TextureDebug> _textureDebug;
    wheels::OwningPtr<DepthOfField> _depthOfField;
    wheels::OwningPtr<ImageBasedLighting> _imageBasedLighting;
    wheels::OwningPtr<TemporalAntiAliasing> _temporalAntiAliasing;
    wheels::OwningPtr<MeshletCuller> _meshletCuller;
    wheels::OwningPtr<TextureReadback> _textureReadback;

    wheels::OwningPtr<Profiler> _profiler;

    bool _useFpsLimit{true};
    int32_t _fpsLimit{140};
    bool _recompileShaders{false};
    bool _referenceRt{false};
    bool _renderDeferred{true};
    bool _deferredRt{false};
    bool _renderDoF{false};
    bool _textureDebugActive{false};
    bool _drawUi{true};
    bool _forceViewportRecreate{false};
    bool _forceCamUpdate{true};
    bool _applyIbl{false};
    bool _sceneChanged{false};
    bool _newSceneDataLoaded{false};
    bool _applyTaa{true};
    bool _applyJitter{true};
    DrawType _drawType{DrawType::Default};

    bool _camFreeLook{false};
    CameraTransform _sceneCameraTransform;
    CameraParameters _cameraParameters;
    wheels::Optional<FrustumCorners> _debugFrustum;
    glm::vec3 _frustumDebugColor{1.f, 1.f, 1.f};
    bool _pickFocusDistance{false};
    glm::vec2 _pickedFocusPx{-1.f, -1.f};
    bool _waitFocusDistance{false};

    wheels::StaticArray<SceneStats, MAX_FRAMES_IN_FLIGHT> _sceneStats;
    wheels::StaticArray<BufferHandle, MAX_FRAMES_IN_FLIGHT> _drawStats;

    std::chrono::high_resolution_clock::time_point _lastTimeChange;
    float _timeOffsetS{0.f};
    bool _isPlaying{false};

    uint32_t _ctorScratchHighWatermark{0};

    Timer _frameTimer;
    std::chrono::time_point<std::chrono::file_clock> _recompileTime;

    wheels::StaticArray<vk::Semaphore, MAX_FRAMES_IN_FLIGHT>
        _imageAvailableSemaphores;
    wheels::StaticArray<vk::Semaphore, MAX_FRAMES_IN_FLIGHT>
        _renderFinishedSemaphores;

    std::future<wheels::HashSet<std::filesystem::path>> _fileChanges;
};

#endif // PROSPER_APP_HPP
