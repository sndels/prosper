#ifndef PROSPER_RENDER_GBUFFER_RENDERER_OUTPUT_HPP
#define PROSPER_RENDER_GBUFFER_RENDERER_OUTPUT_HPP

#include "render/RenderResourceHandle.hpp"

#include <vulkan/vulkan.hpp>

namespace render
{

struct GBuffer
{
    static const vk::Format sAlbedoRoughnessFormat = vk::Format::eR8G8B8A8Unorm;
    static const vk::Format sNormalMetalnessFormat =
        vk::Format::eA2B10G10R10UnormPack32;

    ImageHandle albedoRoughness;
    ImageHandle normalMetalness;
    ImageHandle velocity;
    ImageHandle depth;

    void create(const vk::Extent2D &extent);
    void setHistoryDebugNames() const;
    void releaseAll() const;
    void preserveAll() const;
};

} // namespace render

#endif // PROSPER_RENDER_GBUFFER_RENDERER_OUTPUT_HPP
