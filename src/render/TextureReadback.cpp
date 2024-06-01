#include "TextureReadback.hpp"

#include "../gfx/VkUtils.hpp"
#include "../utils/Profiler.hpp"
#include "RenderResources.hpp"

using namespace glm;
using namespace wheels;

namespace
{

struct PCBlock
{
    vec2 uv{};
};

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/texture_readback.comp",
        .debugName = String{alloc, "TextureReadbackCS"},
    };
}

} // namespace

TextureReadback::~TextureReadback()
{
    // Don't check for _initialized as we might be cleaning up after a failed
    // init.
    gDevice.destroy(_buffer);
}

void TextureReadback::init(
    ScopedScratch scopeAlloc, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc)
{
    WHEELS_ASSERT(!_initialized);
    WHEELS_ASSERT(resources != nullptr);

    _resources = resources;
    _computePass.init(
        WHEELS_MOV(scopeAlloc), staticDescriptorsAlloc,
        shaderDefinitionCallback);
    _buffer = gDevice.createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = sizeof(vec4),
                .usage = vk::BufferUsageFlagBits::eTransferDst,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .debugName = "TextureReadbackHostBuffer",
    });

    _initialized = true;
}

void TextureReadback::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(_initialized);

    _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

void TextureReadback::startFrame()
{
    if (_framesUntilReady > 0)
        _framesUntilReady--;
    else if (_framesUntilReady == 0)
        WHEELS_ASSERT(!"Forgot to call readback() on subsequent frames after "
                       "queueing readback");
}

void TextureReadback::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, ImageHandle inTexture,
    vec2 px, uint32_t nextFrame, Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);
    WHEELS_ASSERT(
        _framesUntilReady == -1 && "Readback already queued and unread");
    WHEELS_ASSERT(profiler != nullptr);

    {
        const BufferHandle deviceReadback = _resources->buffers.create(
            BufferDescription{
                .byteSize = _buffer.byteSize,
                .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eTransferSrc,
                .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            },
            "TextureReadbackDeviceBuffer");

        _computePass.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = _resources->images.resource(inTexture).view,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler = _resources->nearestSampler,
                }},
                DescriptorInfo{vk::DescriptorBufferInfo{
                    .buffer = _resources->buffers.nativeHandle(deviceReadback),
                    .range = VK_WHOLE_SIZE,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), *_resources, cb,
            Transitions{
                .images =
                    StaticArray<ImageTransition, 1>{
                        ImageTransition{
                            inTexture, ImageState::ComputeShaderSampledRead},
                    },
                .buffers =
                    StaticArray<BufferTransition, 1>{
                        BufferTransition{
                            deviceReadback, BufferState::ComputeShaderWrite},
                    },
            });

        const auto _s = profiler->createCpuGpuScope(cb, "TextureReadback");

        const vk::Extent3D inRes =
            _resources->images.resource(inTexture).extent;
        const PCBlock pcBlock{
            .uv = px / vec2(inRes.width, inRes.height),
        };

        const uvec3 extent = uvec3{1u, 1u, 1u};
        const vk::DescriptorSet storageSet = _computePass.storageSet(nextFrame);
        _computePass.record(cb, pcBlock, extent, Span{&storageSet, 1});

        _resources->buffers.transition(
            cb, deviceReadback, BufferState::TransferSrc);
        // We know the host readback buffer is not used this frame so no need
        // for a barrier here

        const vk::BufferCopy region{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = _buffer.byteSize,
        };
        cb.copyBuffer(
            _resources->buffers.nativeHandle(deviceReadback), _buffer.handle, 1,
            &region);

        _resources->buffers.release(deviceReadback);

        _framesUntilReady = MAX_FRAMES_IN_FLIGHT;
    }
}

Optional<vec4> TextureReadback::readback()
{
    WHEELS_ASSERT(_framesUntilReady >= 0 && "No readback in flight");

    if (_framesUntilReady > 0)
        return {};

    _framesUntilReady = -1;
    return *reinterpret_cast<vec4 *>(_buffer.mapped);
}
