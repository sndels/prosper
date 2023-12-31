#include "DeferredLoadingContext.hpp"

#include "../gfx/Device.hpp"
#include "../gfx/VkUtils.hpp"
#include <wheels/allocators/linear_allocator.hpp>
#include <wheels/allocators/scoped_scratch.hpp>

using namespace wheels;

namespace
{

void loadingWorker(DeferredLoadingContext *ctx)
{
    // TODO:
    // Make clang-tidy treat WHEELS_ASSERT as assert so that it considers them
    // valid null checks
    WHEELS_ASSERT(
        ctx != nullptr && ctx->device != nullptr &&
        ctx->device->transferQueue().has_value() &&
        ctx->device->graphicsQueue() != *ctx->device->transferQueue());

    // Enough for 4K textures, it seems
    LinearAllocator scratchBacking{megabytes(256)};
    ScopedScratch scopeAlloc{scratchBacking};

    while (!ctx->interruptLoading)
    {
        if (ctx->workerLoadedImageCount == ctx->gltfModel.images.size())
            break;

        WHEELS_ASSERT(
            ctx->gltfModel.images.size() > ctx->workerLoadedImageCount);
        const tinygltf::Image &image =
            ctx->gltfModel.images[ctx->workerLoadedImageCount];
        if (image.uri.empty())
            throw std::runtime_error("Embedded glTF textures aren't supported. "
                                     "Scene should be glTF + "
                                     "bin + textures.");

        ctx->cb.reset();
        ctx->cb.begin(vk::CommandBufferBeginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        });

        Texture2D tex{
            scopeAlloc.child_scope(),
            ctx->device,
            ctx->sceneDir / image.uri,
            ctx->cb,
            ctx->stagingBuffers[0],
            true,
            true};

        const QueueFamilies &families = ctx->device->queueFamilies();
        WHEELS_ASSERT(families.graphicsFamily.has_value());
        WHEELS_ASSERT(families.transferFamily.has_value());

        if (*families.graphicsFamily != *families.transferFamily)
        {
            const vk::ImageMemoryBarrier2 releaseBarrier{
                .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eNone,
                .dstAccessMask = vk::AccessFlagBits2::eNone,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .srcQueueFamilyIndex = *families.transferFamily,
                .dstQueueFamilyIndex = *families.graphicsFamily,
                .image = tex.nativeHandle(),
                .subresourceRange =
                    vk::ImageSubresourceRange{
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = VK_REMAINING_MIP_LEVELS,
                        .baseArrayLayer = 0,
                        .layerCount = VK_REMAINING_ARRAY_LAYERS,
                    },
            };
            ctx->cb.pipelineBarrier2(vk::DependencyInfo{
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &releaseBarrier,
            });
        }

        ctx->cb.end();

        const vk::Queue transferQueue = *ctx->device->transferQueue();
        const vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &ctx->cb,
        };
        checkSuccess(
            transferQueue.submit(1, &submitInfo, vk::Fence{}),
            "submitTextureUpload");
        // We could have multiple uploads in flight, but let's be simple for now
        transferQueue.waitIdle();

        ctx->workerLoadedImageCount++;

        const uint32_t previousHighWatermark = ctx->allocationHighWatermark;
        if (scratchBacking.allocated_byte_count_high_watermark() >
            previousHighWatermark)
        {
            ctx->allocationHighWatermark = asserted_cast<uint32_t>(
                scratchBacking.allocated_byte_count_high_watermark());
        }

        {
            std::unique_lock lock{ctx->loadedTextureMutex};

            if (ctx->loadedTexture.has_value())
                ctx->loadedTextureTaken.wait(lock);
            WHEELS_ASSERT(!ctx->loadedTexture.has_value());

            ctx->loadedTexture.emplace(WHEELS_MOV(tex));

            lock.unlock();
        }
    }
}

} // namespace

Buffer createTextureStaging(Device *device)
{
    // Assume at most 4k at 8bits per channel
    const vk::DeviceSize stagingSize = static_cast<size_t>(4096) *
                                       static_cast<size_t>(4096) *
                                       sizeof(uint32_t);
    return device->createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = stagingSize,
                .usage = vk::BufferUsageFlagBits::eTransferSrc,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .createMapped = true,
        .debugName = "Texture2DStaging",
    });
}

DeferredLoadingContext::DeferredLoadingContext(
    Allocator &alloc, Device *device, std::filesystem::path sceneDir,
    const tinygltf::Model &gltfModel)
: device{device}
, sceneDir{WHEELS_MOV(sceneDir)}
, gltfModel{gltfModel}
, materials{alloc, gltfModel.materials.size()}
{
    WHEELS_ASSERT(device != nullptr);

    // One of these is used by the worker implementation, all by the
    // single threaded one
    for (uint32_t i = 0; i < stagingBuffers.capacity(); ++i)
        stagingBuffers[i] = createTextureStaging(device);
}

DeferredLoadingContext::~DeferredLoadingContext()
{
    if (device != nullptr)
    {
        if (worker.has_value())
        {
            {
                const std::lock_guard _lock{loadedTextureMutex};
                if (loadedTexture.has_value())
                    const Texture2D _tex = loadedTexture.take();
            }
            loadedTextureTaken.notify_all();

            interruptLoading = true;
            worker->join();
        }

        for (const Buffer &buffer : stagingBuffers)
            device->destroy(buffer);
    }
}

void DeferredLoadingContext::launch()
{
    WHEELS_ASSERT(
        !worker.has_value() && "Tried to launch deferred loading worker twice");

    const Optional<vk::CommandPool> transferPool = device->transferPool();
    if (transferPool.has_value())
    {
        WHEELS_ASSERT(device->transferQueue().has_value());

        cb = device->logical().allocateCommandBuffers(
            vk::CommandBufferAllocateInfo{
                .commandPool = *transferPool,
                .level = vk::CommandBufferLevel::ePrimary,
                .commandBufferCount = 1})[0];
        worker = std::thread{&loadingWorker, this};
    }
}
