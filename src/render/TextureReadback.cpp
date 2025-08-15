#include "TextureReadback.hpp"

#include "gfx/Device.hpp"
#include "render/RenderResources.hpp"
#include "utils/Profiler.hpp"

#include <shader_structs/push_constants/texture_readback.h>

using namespace glm;
using namespace wheels;

namespace render
{

namespace
{

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
    // Don't check for m_initialized as we might be cleaning up after a failed
    // init.
    gfx::gDevice.destroy(m_buffer);
}

void TextureReadback::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(WHEELS_MOV(scopeAlloc), shaderDefinitionCallback);
    m_buffer = gfx::gDevice.createBuffer(gfx::BufferCreateInfo{
        .desc =
            gfx::BufferDescription{
                .byteSize = sizeof(vec4),
                .usage = vk::BufferUsageFlagBits::eTransferDst,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .debugName = "TextureReadbackHostBuffer",
    });

    m_initialized = true;
}

void TextureReadback::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

void TextureReadback::startFrame()
{
    if (m_framesUntilReady > 0)
        m_framesUntilReady--;
    else if (m_framesUntilReady == 0)
        WHEELS_ASSERT(!"Forgot to call readback() on subsequent frames after "
                       "queueing readback");
}

void TextureReadback::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, ImageHandle inTexture,
    vec2 px, uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);
    WHEELS_ASSERT(
        m_framesUntilReady == -1 && "Readback already queued and unread");

    PROFILER_CPU_SCOPE("TextureReadback");

    {
        const BufferHandle deviceReadback = gRenderResources.buffers->create(
            gfx::BufferDescription{
                .byteSize = m_buffer.byteSize,
                .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eTransferSrc,
                .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            },
            "TextureReadbackDeviceBuffer");

        const vk::DescriptorSet storageSet = m_computePass.updateStorageSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(inTexture).view,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler = gRenderResources.nearestSampler,
                }},
                gfx::DescriptorInfo{vk::DescriptorBufferInfo{
                    .buffer =
                        gRenderResources.buffers->nativeHandle(deviceReadback),
                    .range = VK_WHOLE_SIZE,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images =
                    StaticArray<ImageTransition, 1>{
                        ImageTransition{
                            inTexture,
                            gfx::ImageState::ComputeShaderSampledRead},
                    },
                .buffers =
                    StaticArray<BufferTransition, 1>{
                        BufferTransition{
                            deviceReadback,
                            gfx::BufferState::ComputeShaderWrite},
                    },
            });

        PROFILER_GPU_SCOPE(cb, "TextureReadback");

        const vk::Extent3D inRes =
            gRenderResources.images->resource(inTexture).extent;
        const TextureReadbackPC pcBlock{
            .uv = px / vec2(inRes.width, inRes.height),
        };

        const uvec3 groupCount{1u, 1u, 1u};
        m_computePass.record(cb, pcBlock, groupCount, Span{&storageSet, 1});

        gRenderResources.buffers->transition(
            cb, deviceReadback, gfx::BufferState::TransferSrc);
        // We know the host readback buffer is not used this frame so no need
        // for a barrier here

        const vk::BufferCopy region{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = m_buffer.byteSize,
        };
        cb.copyBuffer(
            gRenderResources.buffers->nativeHandle(deviceReadback),
            m_buffer.handle, 1, &region);

        gRenderResources.buffers->release(deviceReadback);

        m_framesUntilReady = MAX_FRAMES_IN_FLIGHT;
    }
}

Optional<vec4> TextureReadback::readback()
{
    WHEELS_ASSERT(m_framesUntilReady >= 0 && "No readback in flight");

    if (m_framesUntilReady > 0)
        return {};

    m_framesUntilReady = -1;
    return *reinterpret_cast<vec4 *>(m_buffer.mapped);
}

} // namespace render
