#ifndef PROSPER_RENDER_TEXTURE_DEBUG_HPP
#define PROSPER_RENDER_TEXTURE_DEBUG_HPP

#include "Allocators.hpp"
#include "gfx/Fwd.hpp"
#include "gfx/Resources.hpp"
#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "utils/ForEach.hpp"
#include "utils/Utils.hpp"

#include <glm/glm.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/hash_map.hpp>
#include <wheels/containers/static_array.hpp>

#define TEXTURE_DEBUG_CHANNEL_TYPES R, G, B, A, RGB

#define TEXTURE_DEBUG_CHANNEL_TYPES_AND_COUNT TEXTURE_DEBUG_CHANNEL_TYPES, Count

#define TEXTURE_DEBUG_CHANNEL_TYPES_STRINGIFY(t) #t,

#define TEXTURE_DEBUG_CHANNEL_TYPES_STRS                                       \
    FOR_EACH(TEXTURE_DEBUG_CHANNEL_TYPES_STRINGIFY, TEXTURE_DEBUG_CHANNEL_TYPES)

class TextureDebug
{
  public:
    enum class ChannelType : uint32_t
    {
        TEXTURE_DEBUG_CHANNEL_TYPES_AND_COUNT
    };

    TextureDebug() noexcept = default;
    ~TextureDebug();

    TextureDebug(const TextureDebug &other) = delete;
    TextureDebug(TextureDebug &&other) = delete;
    TextureDebug &operator=(const TextureDebug &other) = delete;
    TextureDebug &operator=(TextureDebug &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    void drawUi(uint32_t nextFrame);

    [[nodiscard]] ImageHandle record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        vk::Extent2D outSize, wheels::Optional<glm::vec2> cursorCoord,
        uint32_t nextFrame);

    [[nodiscard]] static bool textureSelected();

  private:
    bool m_initialized{false};
    wheels::StaticArray<Buffer, MAX_FRAMES_IN_FLIGHT> m_readbackBuffers;

    ComputePass m_computePass;

    struct TargetSettings
    {
        glm::vec2 range{0.f, 1.f};
        int32_t lod{0};
        ChannelType channelType{ChannelType::RGB};
        bool absBeforeRange{false};
        bool useBilinearSampler{false};
    };
    // TODO:
    // Not a String because type conversions and allocations from StrSpan. Is
    // there a better universal solution?
    wheels::HashMap<uint64_t, TargetSettings> m_targetSettings{
        gAllocators.general};
    bool m_zoom{false};
};

#endif // PROSPER_RENDER_TEXTURE_DEBUG_HPP
