#ifndef PROSPER_APP_HPP
#define PROSPER_APP_HPP

#include "Window.hpp"
#include "gfx/Device.hpp"
#include "gfx/Swapchain.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "scene/Camera.hpp"
#include "scene/Fwd.hpp"
#include "utils/Profiler.hpp"
#include "utils/Timer.hpp"

#include <filesystem>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/allocators/tlsf_allocator.hpp>
#include <wheels/containers/static_array.hpp>

class App
{
  public:
    struct Settings
    {
        std::filesystem::path scene;
        bool deferredLoading{false};
        Device::Settings device;
    };

    App(const Settings &settings);
    ~App();

    App(const App &other) = delete;
    App(App &&other) = delete;
    App &operator=(const App &other) = delete;
    App &operator=(App &&other) = delete;

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
        wheels::ScopedScratch scopeAlloc,
        const wheels::Array<Profiler::ScopeData> &profilerDatas,
        uint32_t scopeHighWatermark);
    void drawOptions();
    void drawRendererSettings(UiChanges &uiChanges);
    void drawProfiling(
        wheels::ScopedScratch scopeAlloc,
        const wheels::Array<Profiler::ScopeData> &profilerDatas);
    void drawMemory(uint32_t scopeHighWatermark);
    // Returns true if time was tweaked
    bool drawTimeline();
    // Returns true if settings changed
    bool drawCameraUi();

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
    void blitColorToFinalComposite(
        vk::CommandBuffer cb, ImageHandle toneMapped);
    void blitFinalComposite(vk::CommandBuffer cb, uint32_t nextImage);
    // Returns true if present succeeded, false if swapchain should be recreated
    [[nodiscard]] bool submitAndPresent(
        vk::CommandBuffer cb, uint32_t nextFrame);
    void handleResizes(
        wheels::ScopedScratch scopeAlloc, bool shouldResizeSwapchain);

    wheels::TlsfAllocator _generalAlloc;

    InputHandler _inputHandler;
    std::unique_ptr<Window>
        _window; // Needs to be valid before and after everything else
    std::unique_ptr<Device>
        _device; // Needs to be valid before and after all other vk resources

    // This allocator should only be used for the descriptors that can live
    // until the end of the program. As such, reset() shouldn't be called so
    // that users can rely on the descriptors being there once allocated.
    std::unique_ptr<DescriptorAllocator> _staticDescriptorsAlloc;

    std::unique_ptr<Swapchain> _swapchain;
    wheels::StaticArray<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT>
        _commandBuffers;

    vk::Extent2D _viewportExtent{};

    // Stored here, managed by (earliest) passes that write to them
    std::unique_ptr<RenderResources> _resources;

    std::unique_ptr<Camera> _cam;
    std::unique_ptr<World> _world;

    std::unique_ptr<LightClustering> _lightClustering;
    std::unique_ptr<ForwardRenderer> _forwardRenderer;
    std::unique_ptr<GBufferRenderer> _gbufferRenderer;
    std::unique_ptr<DeferredShading> _deferredShading;
    std::unique_ptr<RtDirectIllumination> _rtDirectIllumination;
    std::unique_ptr<RtReference> _rtReference;
    std::unique_ptr<SkyboxRenderer> _skyboxRenderer;
    std::unique_ptr<DebugRenderer> _debugRenderer;
    std::unique_ptr<ToneMap> _toneMap;
    std::unique_ptr<ImGuiRenderer> _imguiRenderer;
    std::unique_ptr<TextureDebug> _textureDebug;
    std::unique_ptr<DepthOfField> _depthOfField;
    std::unique_ptr<ImageBasedLighting> _imageBasedLighting;

    std::unique_ptr<Profiler> _profiler;

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

    bool _camFreeLook{false};
    CameraTransform _sceneCameraTransform;
    CameraParameters _cameraParameters;

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
};

#endif // PROSPER_APP_HPP
