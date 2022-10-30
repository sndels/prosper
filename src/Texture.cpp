#include "Texture.hpp"

#include <cmath>
#include <iostream>
#include <variant>

#include <compressonator.h>
#include <gli/gli.hpp>
#include <stb_image.h>
#include <thread>
#include <tiny_gltf.h>

#include "Timer.hpp"
#include "Utils.hpp"

// Can be used to force re-caching for all textures
// #define INVALIDATE_CACHE

namespace
{

constexpr float sCompressionQuality = 0.05f; // Range is [0.05, 1.0]

constexpr const char *sCompressonatorErrorStrs[] = {
    "CMP_OK",
    "CMP_ABORTED",
    "CMP_ERR_INVALID_SOURCE_TEXTURE",
    "CMP_ERR_INVALID_DEST_TEXTURE",
    "CMP_ERR_UNSUPPORTED_SOURCE_FORMAT",
    "CMP_ERR_UNSUPPORTED_DEST_FORMAT",
    "CMP_ERR_UNSUPPORTED_GPU_ASTC_DECODE",
    "CMP_ERR_UNSUPPORTED_GPU_BASIS_DECODE",
    "CMP_ERR_SIZE_MISMATCH",
    "CMP_ERR_UNABLE_TO_INIT_CODEC",
    "CMP_ERR_UNABLE_TO_INIT_DECOMPRESSLIB",
    "CMP_ERR_UNABLE_TO_INIT_COMPUTELIB",
    "CMP_ERR_CMP_DESTINATION",
    "CMP_ERR_MEM_ALLOC_FOR_MIPSET",
    "CMP_ERR_UNKNOWN_DESTINATION_FORMAT",
    "CMP_ERR_FAILED_HOST_SETUP",
    "CMP_ERR_PLUGIN_FILE_NOT_FOUND",
    "CMP_ERR_UNABLE_TO_LOAD_FILE",
    "CMP_ERR_UNABLE_TO_CREATE_ENCODER",
    "CMP_ERR_UNABLE_TO_LOAD_ENCODER",
    "CMP_ERR_NOSHADER_CODE_DEFINED",
    "CMP_ERR_GPU_DOESNOT_SUPPORT_COMPUTE",
    "CMP_ERR_NOPERFSTATS",
    "CMP_ERR_GPU_DOESNOT_SUPPORT_CMP_EXT",
    "CMP_ERR_GAMMA_OUTOFRANGE",
    "CMP_ERR_PLUGIN_SHAREDIO_NOT_SET",
    "CMP_ERR_UNABLE_TO_INIT_D3DX",
    "CMP_FRAMEWORK_NOT_INITIALIZED",
    "CMP_ERR_GENERIC",
};

struct UncompressedPixelData
{
    // 'dataOwned', if not null, needs to be freed with stbi_image_free
    // 'data' may or may not be == 'dataOwned'
    const uint8_t *data{nullptr};
    stbi_uc *dataOwned{nullptr};
    vk::Extent2D extent{0};
    uint32_t channels{0};
};
UncompressedPixelData pixelsFromFile(const std::filesystem::path &path)
{
    const auto pathString = path.string();
    int w = 0;
    int h = 0;
    int channels = 0;
    stbi_uc *pixels = stbi_load(pathString.c_str(), &w, &h, &channels, 0);
    if (pixels == nullptr)
        throw std::runtime_error("Failed to load texture '" + pathString + "'");

    assert(sizeof(uint8_t) == sizeof(stbi_uc));
    return UncompressedPixelData{
        .data = reinterpret_cast<uint8_t *>(pixels),
        .dataOwned = pixels,
        .extent =
            vk::Extent2D{
                asserted_cast<uint32_t>(w),
                asserted_cast<uint32_t>(h),
            },
        .channels = asserted_cast<uint32_t>(channels),
    };
}

// Returns path of the corresponding cached file, creating the folders up to it
// if those don't exists
std::filesystem::path getCachePath(const std::filesystem::path &source)
{
    const auto cacheFolder = source.parent_path() / "prosper_cache";
    if (!std::filesystem::exists(cacheFolder))
        std::filesystem::create_directories(cacheFolder);

    auto cacheFile = source.filename();
    cacheFile.replace_extension("dds");
    return cacheFolder / cacheFile;
}

bool cmpCallback(
    float fProgress, CMP_DWORD_PTR /*pUser1*/, CMP_DWORD_PTR /*pUser2*/)
{
    printf("\r%3.0f", fProgress);
    // Return true to abort compression
    return false;
}

CMP_MipSet getMips(const std::filesystem::path &path)
{
    CMP_ERROR status;
    auto throwStatus = [](const std::string &msg, CMP_ERROR status)
    {
        throw std::runtime_error(
            msg + ": " +
            sCompressonatorErrorStrs[static_cast<uint32_t>(status)]);
    };

    const auto cachePath = getCachePath(path);
#ifndef INVALIDATE_CACHE
    if (!std::filesystem::exists(cachePath))
#endif // INVALIDATE_CACHE
    {
        // TODO: Should move this code to world and process all missing images
        // at once with the parallel cmp interface

        CMP_MipSet srcMips = {};
        status = CMP_LoadTexture(path.string().c_str(), &srcMips);
        if (status != CMP_OK)
            throwStatus("CMP_LoadTexture (source)", status);

        assert(
            (srcMips.m_format == CMP_FORMAT_RGBA_8888 ||
             srcMips.m_format == CMP_FORMAT_RGB_888) &&
            "BC7 only works for RGB or RGBA");

        CMP_INT minMip =
            CMP_CalcMaxMipLevel(srcMips.m_nHeight, srcMips.m_nWidth, true);
        if (minMip > srcMips.m_nMipLevels)
        {
            status =
                static_cast<CMP_ERROR>(CMP_GenerateMIPLevels(&srcMips, minMip));
            if (status != CMP_OK)
            {
                CMP_FreeMipSet(&srcMips);
                throwStatus("CMP_GenerateMipLevels", status);
            }
        }

        // TODO: Need to force linear?
        KernelOptions options = {
            .fquality = sCompressionQuality,
            .format = CMP_FORMAT_BC7,
            .encodeWith = CMP_HPC,
            // Don't saturate the cpu as that seems to cause contention here
            .threads =
                asserted_cast<CMP_INT>(std::thread::hardware_concurrency() - 6),
        };

        CMP_MipSet outMips = {};
        printf("Processing %s\n", path.string().c_str());
        Timer t;
        status = CMP_ProcessTexture(&srcMips, &outMips, options, &cmpCallback);
        if (status != CMP_OK)
        {
            CMP_FreeMipSet(&srcMips);
            throwStatus("CMP_ProcessTexture", status);
        }
        // Compressonator 4.2 leaves its global encoder plugin alive, causing
        // the stack pointer for the first CMP_ProcessTexture options to be used
        // for all further ones. That of course goes to town on unrelated stack
        // when calls are made with non-identical call paths. This can be
        // avoided by manually destroying the plugin
        status = CMP_DestroyComputeLibrary(true);
        if (status != CMP_OK)
        {
            CMP_FreeMipSet(&srcMips);
            throwStatus("CMP_DestroyComputeLibrary", status);
        }
        printf("\r%.2fs\n", t.getSeconds());

        status = CMP_SaveTexture(cachePath.string().c_str(), &outMips);
        CMP_FreeMipSet(&srcMips);
        CMP_FreeMipSet(&outMips);
        if (status != CMP_OK)
            throwStatus("CMP_SaveTexture", status);
    }

    CMP_MipSet outMips = {};
    // Load from file to make sure what we wrote is good
    status = CMP_LoadTexture(cachePath.string().c_str(), &outMips);
    if (status != CMP_OK)
        throwStatus("CMP_LoadTexture (cached)", status);

    return outMips;
}

void transitionImageLayout(
    const vk::CommandBuffer &commandBuffer, const vk::Image &image,
    const vk::ImageSubresourceRange &subresourceRange,
    const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout,
    const vk::AccessFlags srcAccessMask, const vk::AccessFlags dstAccessMask,
    const vk::PipelineStageFlags srcStageMask,
    const vk::PipelineStageFlags dstStageMask)

{
    const vk::ImageMemoryBarrier barrier{
        .srcAccessMask = srcAccessMask,
        .dstAccessMask = dstAccessMask,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = subresourceRange,
    };
    commandBuffer.pipelineBarrier(
        srcStageMask, dstStageMask, vk::DependencyFlags{}, 0, nullptr, 0,
        nullptr, 1, &barrier);
}

Buffer stagePixels(
    Device *device, const void *data, vk::Extent2D extent,
    uint32_t bytesPerPixel)
{
    const vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(extent.width) *
                                     extent.height * bytesPerPixel;

    const Buffer stagingBuffer = device->createBuffer(BufferCreateInfo{
        .byteSize = imageSize,
        .usage = vk::BufferUsageFlagBits::eTransferSrc,
        .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                      vk::MemoryPropertyFlagBits::eHostCoherent,
        .createMapped = true,
        .debugName = "Texture2DStaging",
    });

    memcpy(stagingBuffer.mapped, data, asserted_cast<size_t>(imageSize));

    return stagingBuffer;
}

} // namespace

Texture::Texture(Device *device)
: _device{device}
{
}

Texture::~Texture() { destroy(); }

Texture::Texture(Texture &&other) noexcept
: _device{other._device}
, _image{other._image}
{
    other._device = nullptr;
}

Texture &Texture::operator=(Texture &&other) noexcept
{
    if (this != &other)
    {
        destroy();
        _device = other._device;
        _image = other._image;

        other._device = nullptr;
    }
    return *this;
}

void Texture::destroy()
{
    if (_device != nullptr)
        _device->destroy(_image);
}

Texture2D::Texture2D(Device *device, const std::filesystem::path &path)
: Texture(device)
{
    auto mips = getMips(path);

    assert(mips.m_format == CMP_FORMAT_BC7);

    const auto width = asserted_cast<uint32_t>(mips.m_nWidth);
    const auto height = asserted_cast<uint32_t>(mips.m_nHeight);
    const auto levels = asserted_cast<uint32_t>(mips.m_nMipLevels);

    // Both transfer source and destination as pixels will be transferred to it
    // and mipmaps will be generated from it
    _image = _device->createImage(ImageCreateInfo{
        .format = vk::Format::eBc7UnormBlock,
        .width = width,
        .height = height,
        .mipCount = levels,
        .layerCount = 1,
        .usageFlags = vk::ImageUsageFlagBits::eTransferSrc |
                      vk::ImageUsageFlagBits::eTransferDst |
                      vk::ImageUsageFlagBits::eSampled,
        .debugName = "Texture2D",
    });

    const auto rawByteSize = device->rawByteSize(_image);

    const Buffer stagingBuffer = device->createBuffer(BufferCreateInfo{
        .byteSize = rawByteSize,
        .usage = vk::BufferUsageFlagBits::eTransferSrc,
        .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                      vk::MemoryPropertyFlagBits::eHostCoherent,
        .createMapped = true,
        .debugName = "Texture2DStaging",
    });

    memcpy(stagingBuffer.mapped, mips.pData, mips.dwDataSize);

    std::vector<vk::BufferImageCopy> regions;
    regions.reserve(levels);
    size_t writeOffset = 0;
    for (auto i = 0; i < mips.m_nMipLevels; ++i)
    {
        const auto w = std::max(width >> i, 1u);
        const auto h = std::max(height >> i, 1u);
        regions.push_back(vk::BufferImageCopy{
            .bufferOffset = writeOffset,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource =
                vk::ImageSubresourceLayers{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = asserted_cast<uint32_t>(i),
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            .imageOffset = {0, 0, 0},
            .imageExtent = {w, h, 1u},
        });

        // TODO: Calculate offset to mip
        assert(mips.m_format == CMP_FORMAT_BC7);
        // BC7 is 128b for 4x4
        const auto pixels = w * h;
        const auto blocks = (pixels - 1) / 16 + 1;
        const auto offset = blocks * 16;
        writeOffset += offset;
    }

    const auto commandBuffer = _device->beginGraphicsCommands();

    _image.transition(
        commandBuffer, ImageState{
                           .stageMask = vk::PipelineStageFlagBits2::eTransfer,
                           .accessMask = vk::AccessFlagBits2::eTransferWrite,
                           .layout = vk::ImageLayout::eTransferDstOptimal,
                       });

    commandBuffer.copyBufferToImage(
        stagingBuffer.handle, _image.handle,
        vk::ImageLayout::eTransferDstOptimal,
        asserted_cast<uint32_t>(regions.size()), regions.data());

    _image.transition(
        commandBuffer,
        ImageState{
            .stageMask = vk::PipelineStageFlagBits2::eFragmentShader |
                         vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
            .accessMask = vk::AccessFlagBits2::eShaderRead,
            .layout = vk::ImageLayout::eShaderReadOnlyOptimal,
        });

    _device->endGraphicsCommands(commandBuffer);

    _device->destroy(stagingBuffer);
    CMP_FreeMipSet(&mips);
}

Texture2D::Texture2D(
    Device *device, const tinygltf::Image &image, const bool mipmap)
: Texture(device)
{
    // TODO: support
    if (image.pixel_type != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
        throw std::runtime_error("Unsupported glTF pixel_type");

    UncompressedPixelData pixels{
        .extent =
            vk::Extent2D{
                asserted_cast<uint32_t>(image.width),
                asserted_cast<uint32_t>(image.height),
            },
    };
    std::vector<uint8_t> tmpPixels;
    if (image.component < 3)
        throw std::runtime_error("Image with less than 3 components");

    if (image.component == 3)
    {
        // Add fourth channel as 3 channel optimal tiling is rarely supported
        tmpPixels.resize(asserted_cast<size_t>(image.width) * image.height * 4);
        const auto *rgb = image.image.data();
        auto *rgba = tmpPixels.data();
        for (int i = 0; i < image.width * image.height; ++i)
        {
            rgba[0] = rgb[0];
            rgba[1] = rgb[1];
            rgba[2] = rgb[2];
            rgb += 3;
            rgba += 4;
        }
        pixels.data = tmpPixels.data();
        pixels.channels = 4;
    }
    else
        pixels.data = reinterpret_cast<const uint8_t *>(image.image.data());

    const auto stagingBuffer =
        stagePixels(_device, pixels.data, pixels.extent, pixels.channels);

    const uint32_t mipLevels =
        mipmap ? asserted_cast<uint32_t>(floor(log2(
                     std::max(pixels.extent.width, pixels.extent.height)))) +
                     1
               : 1;

    createImage(
        stagingBuffer, ImageCreateInfo{
                           .format = vk::Format::eR8G8B8A8Unorm,
                           .width = pixels.extent.width,
                           .height = pixels.extent.height,
                           .mipCount = mipLevels,
                           .layerCount = 1,
                           .usageFlags = vk::ImageUsageFlagBits::eTransferSrc |
                                         vk::ImageUsageFlagBits::eTransferDst |
                                         vk::ImageUsageFlagBits::eSampled,
                           .debugName = "Texture2D",
                       });

    _device->destroy(stagingBuffer);
}

vk::DescriptorImageInfo Texture2D::imageInfo() const
{
    return vk::DescriptorImageInfo{
        .imageView = _image.view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
}

void Texture2D::createImage(
    const Buffer &stagingBuffer, const ImageCreateInfo &info)
{
    // Both transfer source and destination as pixels will be transferred to it
    // and mipmaps will be generated from it
    _image = _device->createImage(info);

    const auto commandBuffer = _device->beginGraphicsCommands();

    _image.transition(
        commandBuffer, ImageState{
                           .stageMask = vk::PipelineStageFlagBits2::eTransfer,
                           .accessMask = vk::AccessFlagBits2::eTransferWrite,
                           .layout = vk::ImageLayout::eTransferDstOptimal,
                       });

    const vk::BufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
            vk::ImageSubresourceLayers{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = _image.extent,
    };
    commandBuffer.copyBufferToImage(
        stagingBuffer.handle, _image.handle,
        vk::ImageLayout::eTransferDstOptimal, 1, &region);

    _device->endGraphicsCommands(commandBuffer);

    createMipmaps(
        vk::Extent2D{
            .width = info.width,
            .height = info.height,
        },
        info.mipCount);
}

void Texture2D::createMipmaps(
    const vk::Extent2D &extent, const uint32_t mipLevels)
{
    // TODO: Check that the texture format supports linear filtering
    const auto buffer = _device->beginGraphicsCommands();

    vk::ImageSubresourceRange subresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };

    int32_t mipWidth = extent.width;
    int32_t mipHeight = extent.height;
    for (uint32_t i = 1; i < mipLevels; ++i)
    {
        // Make sure last operation finished and source is transitioned
        subresourceRange.baseMipLevel = i - 1;
        transitionImageLayout(
            buffer, _image.handle, subresourceRange,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eTransferSrcOptimal,
            vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eTransferRead,
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eTransfer);

        vk::ImageBlit blit{
            .srcSubresource =
                vk::ImageSubresourceLayers{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = i - 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            .srcOffsets = {{
                vk::Offset3D{0},
                vk::Offset3D{mipWidth, mipHeight, 1},
            }},
            .dstSubresource =
                vk::ImageSubresourceLayers{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = i,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            .dstOffsets = {{
                vk::Offset3D{0},
                vk::Offset3D{
                    mipWidth > 1 ? mipWidth / 2 : 1,
                    mipHeight > 1 ? mipHeight / 2 : 1, 1},
            }},
        };
        buffer.blitImage(
            _image.handle, vk::ImageLayout::eTransferSrcOptimal, _image.handle,
            vk::ImageLayout::eTransferDstOptimal, 1, &blit,
            vk::Filter::eLinear);

        // Source needs to be transitioned to shader read optimal
        transitionImageLayout(
            buffer, _image.handle, subresourceRange,
            vk::ImageLayout::eTransferSrcOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderRead,
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader |
                vk::PipelineStageFlagBits::eRayTracingShaderKHR);

        if (mipWidth > 1)
            mipWidth /= 2;
        if (mipHeight > 1)
            mipHeight /= 2;
    }

    // Last mip level needs to be transitioned to shader read optimal
    subresourceRange.baseMipLevel = mipLevels - 1;
    transitionImageLayout(
        buffer, _image.handle, subresourceRange,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader |
            vk::PipelineStageFlagBits::eRayTracingShaderKHR);

    // We went around the state management since it doesn't support
    // per-subresource barriers
    _image.state = ImageState{
        .stageMask = vk::PipelineStageFlagBits2::eFragmentShader |
                     vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
        .accessMask = vk::AccessFlagBits2::eShaderRead,
        .layout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    _device->endGraphicsCommands(buffer);
}

TextureCubemap::TextureCubemap(
    Device *device, const std::filesystem::path &path)
: Texture(device)
{
    const gli::texture_cube cube(gli::load(path.string()));
    assert(!cube.empty());
    assert(cube.faces() == 6);

    const auto mipLevels = asserted_cast<uint32_t>(cube.levels());

    _image = _device->createImage(ImageCreateInfo{
        .format = vk::Format::eR16G16B16A16Sfloat,
        .width = asserted_cast<uint32_t>(cube.extent().x),
        .height = asserted_cast<uint32_t>(cube.extent().y),
        .mipCount = mipLevels,
        .layerCount = 6,
        .createFlags = vk::ImageCreateFlagBits::eCubeCompatible,
        .usageFlags = vk::ImageUsageFlagBits::eTransferDst |
                      vk::ImageUsageFlagBits::eSampled,
        .debugName = "TextureCubemap",
    });

    copyPixels(cube, _image.subresourceRange);

    _sampler = _device->logical().createSampler(vk::SamplerCreateInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .anisotropyEnable = VK_TRUE,
        .maxAnisotropy = 16,
        .minLod = 0,
        .maxLod = static_cast<float>(mipLevels),
    });
}

TextureCubemap::~TextureCubemap()
{
    if (_device != nullptr)
        _device->logical().destroy(_sampler);
}

TextureCubemap::TextureCubemap(TextureCubemap &&other) noexcept
: Texture{std::move(other)}
, _sampler{other._sampler}
{
}

TextureCubemap &TextureCubemap::operator=(TextureCubemap &&other) noexcept
{

    if (this != &other)
    {
        destroy();
        _device = other._device;
        _image = other._image;
        _sampler = other._sampler;

        other._device = nullptr;
    }
    return *this;
}

vk::DescriptorImageInfo TextureCubemap::imageInfo() const
{
    return vk::DescriptorImageInfo{
        .sampler = _sampler,
        .imageView = _image.view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
}

void TextureCubemap::copyPixels(
    const gli::texture_cube &cube,
    const vk::ImageSubresourceRange &subresourceRange) const
{
    const Buffer stagingBuffer = _device->createBuffer(BufferCreateInfo{
        .byteSize = cube.size(),
        .usage = vk::BufferUsageFlagBits::eTransferSrc,
        .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                      vk::MemoryPropertyFlagBits::eHostCoherent,
        .createMapped = true,
        .debugName = "TextureCubemapStaging",
    });

    memcpy(stagingBuffer.mapped, cube.data(), cube.size());

    // Collect memory regions of all faces and their miplevels so their
    // transfers can be submitted together
    const std::vector<vk::BufferImageCopy> regions = [&]
    {
        std::vector<vk::BufferImageCopy> regions;
        size_t offset = 0;
        for (uint32_t face = 0; face < cube.faces(); ++face)
        {
            for (uint32_t mipLevel = 0; mipLevel < cube.levels(); ++mipLevel)
            {
                // Cubemap data contains each face and its miplevels in order
                regions.push_back(vk::BufferImageCopy{
                    .bufferOffset = offset,
                    .bufferRowLength = 0,
                    .bufferImageHeight = 0,
                    .imageSubresource =
                        vk::ImageSubresourceLayers{
                            .aspectMask = vk::ImageAspectFlagBits::eColor,
                            .mipLevel = mipLevel,
                            .baseArrayLayer = face,
                            .layerCount = 1,
                        },
                    .imageOffset = vk::Offset3D{0},
                    .imageExtent =
                        vk::Extent3D{
                            asserted_cast<uint32_t>(
                                cube[face][mipLevel].extent().x),
                            asserted_cast<uint32_t>(
                                cube[face][mipLevel].extent().y),
                            1},
                });
                offset += cube[face][mipLevel].size();
            }
        }
        return regions;
    }();

    const auto copyBuffer = _device->beginGraphicsCommands();

    transitionImageLayout(
        copyBuffer, _image.handle, subresourceRange,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
        vk::AccessFlags{}, vk::AccessFlagBits::eTransferWrite,
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer);

    copyBuffer.copyBufferToImage(
        stagingBuffer.handle, _image.handle,
        vk::ImageLayout::eTransferDstOptimal,
        asserted_cast<uint32_t>(regions.size()), regions.data());

    transitionImageLayout(
        copyBuffer, _image.handle, subresourceRange,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader);

    _device->endGraphicsCommands(copyBuffer);

    _device->destroy(stagingBuffer);
}
