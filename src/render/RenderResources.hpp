#ifndef PROSPER_RENDER_RESOURCES_HPP
#define PROSPER_RENDER_RESOURCES_HPP

#include "gfx/Resources.hpp"
#include "render/RenderBufferCollection.hpp"
#include "render/RenderImageCollection.hpp"
#include "render/RenderTexelBufferCollection.hpp"
#include "scene/DebugGeometry.hpp"
#include "utils/Utils.hpp"

#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/concepts.hpp>
#include <wheels/containers/inline_array.hpp>
#include <wheels/containers/static_array.hpp>
#include <wheels/owning_ptr.hpp>

using ImageTransition = wheels::Pair<ImageHandle, ImageState>;
using BufferTransition = wheels::Pair<BufferHandle, BufferState>;
using TexelBufferTransition = wheels::Pair<TexelBufferHandle, BufferState>;

class RenderResources
{
  public:
    // Both alloc and device need to live as long as this
    RenderResources() noexcept = default;
    ~RenderResources();

    RenderResources(RenderResources &) = delete;
    RenderResources(RenderResources &&) = delete;
    RenderResources &operator=(RenderResources &) = delete;
    RenderResources &operator=(RenderResources &&) = delete;

    void init();
    void destroy();

    // Should be called at the start of the frame so resources will get the
    // correct names set
    void startFrame();

    // Should be called e.g. when viewport is resized since the render resources
    // will be created with different sizes on the next frame
    void destroyResources();

    // Ptrs to control lifetime with init()/destroy()
    wheels::OwningPtr<RenderImageCollection> images;
    wheels::OwningPtr<RenderTexelBufferCollection> texelBuffers;
    wheels::OwningPtr<RenderBufferCollection> buffers;

    vk::Sampler nearestBorderBlackFloatSampler;
    vk::Sampler nearestSampler;
    vk::Sampler bilinearSampler;
    vk::Sampler bilinearBorderTransparentBlackSampler;
    vk::Sampler trilinearSampler;

    // One lines buffer per frame to leave mapped
    wheels::StaticArray<DebugLines, MAX_FRAMES_IN_FLIGHT> debugLines;

  private:
    bool m_initialized{false};
};

// This is depended on by Device and init()/destroy() order relative to other
// similar globals is handled in main()
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern RenderResources gRenderResources;

struct Transitions
{
    wheels::Span<const ImageTransition> images;
    wheels::Span<const BufferTransition> buffers;
    wheels::Span<const TexelBufferTransition> texelBuffers;
};
void transition(
    wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    const Transitions &transitions);

#endif // PROSPER_RENDER_RESOURCES_HPP
