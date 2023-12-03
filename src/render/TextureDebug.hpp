#ifndef PROSPER_RENDER_TEXTURE_DEBUG_HPP
#define PROSPER_RENDER_TEXTURE_DEBUG_HPP

#include <glm/detail/type_vec2.hpp>
#include <glm/fwd.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/hash_map.hpp>
#include <wheels/containers/static_array.hpp>

#include "../gfx/Fwd.hpp"
#include "../utils/Fwd.hpp"
#include "../utils/Utils.hpp"
#include "ComputePass.hpp"
#include "Fwd.hpp"
#include "RenderResourceHandle.hpp"

#include "../utils/ForEach.hpp"

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

    TextureDebug(
        wheels::Allocator &alloc, wheels::ScopedScratch scopeAlloc,
        Device *device, RenderResources *resources,
        DescriptorAllocator *staticDescriptorsAlloc);

    ~TextureDebug() = default;

    TextureDebug(const TextureDebug &other) = delete;
    TextureDebug(TextureDebug &&other) = delete;
    TextureDebug &operator=(const TextureDebug &other) = delete;
    TextureDebug &operator=(TextureDebug &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    void drawUi();

    [[nodiscard]] ImageHandle record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        vk::Extent2D outSize, uint32_t nextFrame, Profiler *profiler);

  private:
    ImageHandle createOutput(vk::Extent2D size);

    RenderResources *_resources{nullptr};

    ComputePass _computePass;

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
    wheels::HashMap<uint64_t, TargetSettings> _targetSettings;
};

#endif // PROSPER_RENDER_TEXTURE_DEBUG_HPP
