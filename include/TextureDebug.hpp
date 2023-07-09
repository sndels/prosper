#ifndef PROSPER_TEXTURE_DEBUG_HPP
#define PROSPER_TEXTURE_DEBUG_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

#include "Device.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "Utils.hpp"

#include "ForEach.hpp"

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
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources,
        DescriptorAllocator *staticDescriptorsAlloc);
    ~TextureDebug();

    TextureDebug(const TextureDebug &other) = delete;
    TextureDebug(TextureDebug &&other) = delete;
    TextureDebug &operator=(const TextureDebug &other) = delete;
    TextureDebug &operator=(TextureDebug &&other) = delete;

    void recompileShaders(wheels::ScopedScratch scopeAlloc);

    void drawUi();

    [[nodiscard]] ImageHandle record(
        vk::CommandBuffer cb, vk::Extent2D outSize, uint32_t nextFrame,
        Profiler *profiler);

  private:
    bool compileShaders(wheels::ScopedScratch scopeAlloc);

    void destroyPipelines();

    void createDescriptorSets(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc);
    void createPipelines();

    ImageHandle createOutput(vk::Extent2D size);

    struct BoundImages
    {
        ImageHandle inColor;
        ImageHandle outColor;
    };
    void updateDescriptorSet(uint32_t nextFrame, const BoundImages &images);

    void recordBarriers(vk::CommandBuffer cb, const BoundImages &images) const;

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    vk::ShaderModule _compSM;
    wheels::Optional<ShaderReflection> _shaderReflection;

    vk::DescriptorSetLayout _descriptorSetLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
        _descriptorSets{{}};
    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
    vk::Sampler _linearSampler;
    vk::Sampler _nearestSampler;

    glm::vec2 _range{0.f, 1.f};
    int32_t _lod{0};
    ChannelType _channelType{ChannelType::RGB};
    bool _absBeforeRange{false};
    bool _useLinearSampler{false};
};

#endif // PROSPER_TEXTURE_DEBUG_HPP
