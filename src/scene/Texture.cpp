#include "Texture.hpp"

#include "gfx/Device.hpp"
#include "utils/Dds.hpp"
#include "utils/Ktx.hpp"
#include "utils/Logger.hpp"
#include "utils/Utils.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <ispc_texcomp.h>
#include <stb_image.h>
#include <stb_image_resize2.h>
#include <wheels/containers/array.hpp>
#include <wheels/containers/pair.hpp>
#include <wyhash.h>

using namespace wheels;

namespace
{

const uint64_t sTextureCacheMagic = 0x5845'5452'5053'5250; // PRSPRTEX
// This should be incremented when changes are made to what's cached
const uint32_t sTextureCacheVersion = 5;

struct UncompressedPixelData
{
    Span<const uint8_t> data;
    vk::Extent2D extent{0};
    uint32_t channels{0};
};

std::filesystem::path cachePath(const std::filesystem::path &source)
{
    const auto cacheFolder = source.parent_path() / "prosper_cache";
    if (!std::filesystem::exists(cacheFolder))
        std::filesystem::create_directory(cacheFolder, source.parent_path());

    auto cacheFile = source.filename();
    cacheFile.replace_extension("dds");
    return cacheFolder / cacheFile;
}

std::filesystem::path cacheTagPath(const std::filesystem::path &cacheFile)
{
    std::filesystem::path tagPath = cacheFile;
    tagPath.replace_extension("prosper_cache_tag");
    return tagPath;
}

struct CacheTag
{
    uint32_t version{0xFFFF'FFFFu};
    // Use write time instead of a hash because hashing a 4k texture is _slow_
    // in debug.
    std::filesystem::file_time_type sourceWriteTime;
};

CacheTag readCacheTag(const std::filesystem::path &cacheFile)
{
    CacheTag tag;

    const std::filesystem::path tagPath = cacheTagPath(cacheFile);
    if (!std::filesystem::exists(tagPath))
        return tag;

    // NOTE:
    // Caches aren't supposed to be portable so we don't pay attention to
    // endianness.
    std::ifstream tagFile{tagPath};

    readRaw(tagFile, tag.version);
    if (sTextureCacheVersion != tag.version)
        return tag;

    // Magic after version because the first two versions didn't have a magic
    // number at all
    uint64_t magic{0};
    static_assert(sizeof(magic) == sizeof(sTextureCacheMagic));

    readRaw(tagFile, magic);
    if (magic != sTextureCacheMagic)
        throw std::runtime_error(
            "Expected a valid texture cache tag in file '" + tagPath.string() +
            "'");

    readRaw(tagFile, tag.sourceWriteTime);

    return tag;
}

void writeCacheTag(
    const std::filesystem::path &cacheFile,
    const std::filesystem::file_time_type &sourceWriteTime)
{
    const std::filesystem::path tagPath = cacheTagPath(cacheFile);

    std::filesystem::remove(tagPath);

    // Write into a tmp file and rename when done to minimize the potential for
    // corrupted files
    std::filesystem::path tagTmpPath = tagPath;
    tagTmpPath.replace_extension("prosper_cache_tag_TMP");

    // NOTE:
    // Caches aren't supposed to be portable so we don't pay attention to
    // endianness.
    std::ofstream tagFile{tagTmpPath, std::ios_base::binary};
    writeRaw(tagFile, sTextureCacheVersion);
    writeRaw(tagFile, sTextureCacheMagic);
    writeRaw(tagFile, sourceWriteTime);
    tagFile.close();

    // Make sure we have rw permissions for the user to be nice
    const std::filesystem::perms initialPerms =
        std::filesystem::status(tagTmpPath).permissions();
    std::filesystem::permissions(
        tagTmpPath, initialPerms | std::filesystem::perms::owner_read |
                        std::filesystem::perms::owner_write);

    // Rename when the file is done to minimize the potential of a corrupted
    // file
    std::filesystem::rename(tagTmpPath, tagPath);
}

bool cacheValid(
    const std::filesystem::path &cacheFile,
    const std::filesystem::file_time_type &sourceWriteTime)
{
    try
    {
        if (!std::filesystem::exists(cacheFile))
        {
            LOG_INFO("Missing cache file %s", cacheFile.string().c_str());
            return false;
        }

        const CacheTag storedTag = readCacheTag(cacheFile);

        if (sTextureCacheVersion != storedTag.version)
        {
            LOG_INFO(
                "Old cache data version for %s", cacheFile.string().c_str());
            return false;
        }

        if (storedTag.sourceWriteTime != sourceWriteTime)
        {
            LOG_INFO("Stale cache for %s", cacheFile.string().c_str());
            return false;
        }
    }
    catch (std::exception &)
    {
        return false;
    }

    return true;
}

void generateMipLevels(
    Array<uint8_t> &rawLevels, Array<uint32_t> &rawLevelByteOffsets,
    const UncompressedPixelData &pixels, TextureColorSpace colorSpace)
{
    const size_t mipLevelCount = rawLevelByteOffsets.size();
    // TODO: Non-8bit channels?
    const uint32_t pixelStride = pixels.channels;
    // TODO: Pass in actual layout
    const auto pixelLayout = static_cast<stbir_pixel_layout>(pixels.channels);

    rawLevelByteOffsets[0] = 0;
    for (uint32_t level = 1; level < mipLevelCount; ++level)
    {
        const uint32_t parentWidth =
            std::max(pixels.extent.width >> (level - 1u), 1u);
        const uint32_t parentHeight =
            std::max(pixels.extent.height >> (level - 1u), 1u);
        const uint32_t width = std::max(pixels.extent.width >> level, 1u);
        const uint32_t height = std::max(pixels.extent.height >> level, 1u);

        const uint32_t parentRowStride = parentWidth * pixelStride;

        rawLevelByteOffsets[level] =
            rawLevelByteOffsets[level - 1] + parentHeight * parentRowStride;

        const uint8_t *parentData = reinterpret_cast<uint8_t *>(
            rawLevels.data() + rawLevelByteOffsets[level - 1]);
        uint8_t *data = reinterpret_cast<uint8_t *>(
            rawLevels.data() + rawLevelByteOffsets[level]);
        if (colorSpace == TextureColorSpace::sRgb)
            stbir_resize_uint8_srgb(
                parentData, asserted_cast<int>(parentWidth),
                asserted_cast<int>(parentHeight), 0, data,
                asserted_cast<int>(width), asserted_cast<int>(height), 0,
                pixelLayout);
        else
        {
            WHEELS_ASSERT(colorSpace == TextureColorSpace::Linear);
            stbir_resize_uint8_linear(
                parentData, asserted_cast<int>(parentWidth),
                asserted_cast<int>(parentHeight), 0, data,
                asserted_cast<int>(width), asserted_cast<int>(height), 0,
                pixelLayout);
        }
    }
}

void compress(
    ScopedScratch scopeAlloc, const std::filesystem::path &targetPath,
    const UncompressedPixelData &pixels, const Texture2DOptions &options)
{
    // First calculate mip count down to 1x1
    const int32_t fullMipLevelCount =
        options.generateMipMaps
            ? asserted_cast<int32_t>(getMipCount(
                  std::max(pixels.extent.width, pixels.extent.height)))
            : 1;
    // Truncate to 4x4 for the final level
    const uint32_t mipLevelCount =
        asserted_cast<uint32_t>(std::max(fullMipLevelCount - 2, 1));

    DxgiFormat format = DxgiFormat::BC7Unorm;
    // All BC7 levels have to divide evenly by 4 in both directions
    for (uint32_t i = 0; i < mipLevelCount; ++i)
    {
        if (std::max(pixels.extent.width >> i, 1u) % 4 != 0 ||
            std::max(pixels.extent.height >> i, 1u) % 4 != 0)
        {
            format = DxgiFormat::R8G8B8A8Unorm;
            break;
        }
    }

    Dds dds{scopeAlloc, pixels.extent.width, pixels.extent.height, 1,
            format,     mipLevelCount};

    Array<uint8_t> rawLevels{scopeAlloc};
    // Twice the size of the first level should be plenty for mips
    rawLevels.resize(pixels.data.size() * 2);

    memcpy(rawLevels.data(), pixels.data.data(), pixels.data.size());

    Array<uint32_t> rawLevelByteOffsets{scopeAlloc};
    rawLevelByteOffsets.resize(mipLevelCount, 0);

    if (mipLevelCount > 1)
        generateMipLevels(
            rawLevels, rawLevelByteOffsets, pixels, options.colorSpace);

    if (format == DxgiFormat::BC7Unorm)
    {
        bc7_enc_settings bc7Settings{};
        // Don't really care about quality at this point, this is much faster
        // than even veryfast
        GetProfile_alpha_ultrafast(&bc7Settings);

        for (uint32_t i = 0; i < mipLevelCount; ++i)
        {

            const uint32_t width = std::max(dds.width >> i, 1u);
            const uint32_t height = std::max(dds.height >> i, 1u);
            WHEELS_ASSERT(
                width % 4 == 0 && height % 4 == 0 &&
                "BC7 mips should be divide evenly by 4x4");
            WHEELS_ASSERT(
                width >= 4 && height >= 4 &&
                "BC7 mip dimensions should be at least 4x4");

            const rgba_surface rgbaSurface{
                .ptr = rawLevels.data() + rawLevelByteOffsets[i],
                .width = asserted_cast<int32_t>(width),
                .height = asserted_cast<int32_t>(height),
                .stride = asserted_cast<int32_t>(width * 4),
            };
            uint8_t *dst = dds.data.data() + dds.levelByteOffsets[i];
            // TODO:
            // 4k textures are really slow to convert even when AVX2 is
            // available. The vectorized code path seems to be used as expected
            // so this is apparently just slow. Parallelize this with OpenMP or
            // something?
            CompressBlocksBC7(&rgbaSurface, dst, &bc7Settings);
        }
    }
    else
    {
        WHEELS_ASSERT(dds.data.size() <= rawLevels.size());
        memcpy(dds.data.data(), rawLevels.data(), dds.data.size());
    }

    writeDds(dds, targetPath);
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

vk::Format asVkFormat(DxgiFormat format)
{
    switch (format)
    {
    case DxgiFormat::R8G8B8A8Unorm:
        return vk::Format::eR8G8B8A8Unorm;
    case DxgiFormat::R9G9B9E5SharedExp:
        return vk::Format::eE5B9G9R9UfloatPack32;
    case DxgiFormat::BC7Unorm:
        return vk::Format::eBc7UnormBlock;
    default:
        break;
    }
    throw std::runtime_error("Unkown DxgiFormat");
}

} // namespace

Texture::~Texture() { destroy(); }

Texture::Texture(Texture &&other) noexcept
: m_image{WHEELS_MOV(other.m_image)}
{
    other.m_image = Image{};
}

Texture &Texture::operator=(Texture &&other) noexcept
{
    if (this != &other)
    {
        destroy();

        m_image = WHEELS_MOV(other.m_image);

        other.m_image = Image{};
    }
    return *this;
}

vk::Image Texture::nativeHandle() const
{
    WHEELS_ASSERT(m_image.handle);
    return m_image.handle;
}

void Texture::destroy() { gDevice.destroy(m_image); }

void Texture2D::init(
    ScopedScratch scopeAlloc, const std::filesystem::path &path,
    vk::CommandBuffer cb, const Buffer &stagingBuffer,
    const Texture2DOptions &options)
{
    const std::filesystem::file_time_type sourceWriteTime =
        std::filesystem::last_write_time(path);

    const auto cached = cachePath(path);
    if (!cacheValid(cached, sourceWriteTime))
    {
        const auto pathString = path.string();
        int width = 0;
        int height = 0;
        int channels = 0;
        const int desiredChannels = 4;
        stbi_uc *stb_pixels = stbi_load(
            pathString.c_str(), &width, &height, &channels, desiredChannels);
        if (stb_pixels == nullptr)
            throw std::runtime_error(
                "Failed to load texture '" + pathString + "'");
        channels = desiredChannels;

        defer { stbi_image_free(stb_pixels); };

        const UncompressedPixelData pixels{
            .data =
                Span{
                    stb_pixels, asserted_cast<size_t>(width) *
                                    asserted_cast<size_t>(height) *
                                    asserted_cast<size_t>(channels)},
            .extent =
                vk::Extent2D{
                    asserted_cast<uint32_t>(width),
                    asserted_cast<uint32_t>(height),
                },
            .channels = asserted_cast<uint32_t>(channels),
        };

        compress(scopeAlloc.child_scope(), cached, pixels, options);

        writeCacheTag(cached, sourceWriteTime);
    }

    // TODO:
    // If cache was invalid, the newly cached one directly from memory
    Dds dds = readDds(scopeAlloc, cached);

    WHEELS_ASSERT(!dds.data.empty());

    const vk::Extent2D extent{
        dds.width,
        dds.height,
    };

    WHEELS_ASSERT(stagingBuffer.mapped != nullptr);
    WHEELS_ASSERT(dds.data.size() <= stagingBuffer.byteSize);

    memcpy(stagingBuffer.mapped, dds.data.data(), dds.data.size());

    // TODO:
    // Use srgb formats in dds for srgb data, have a flag in texture ctor for
    // using unorm for snorm inputs. Having srgb data say it's unorm is
    // potentially confusing.

    const std::filesystem::path relPath = relativePath(path);

    m_image = gDevice.createImage(ImageCreateInfo{
        .desc =
            ImageDescription{
                .format = asVkFormat(dds.format),
                .width = extent.width,
                .height = extent.height,
                .mipCount = dds.mipLevelCount,
                .layerCount = 1,
                .usageFlags = vk::ImageUsageFlagBits::eTransferSrc |
                              vk::ImageUsageFlagBits::eTransferDst |
                              vk::ImageUsageFlagBits::eSampled,
            },
        .debugName = relPath.generic_string().c_str(),
    });

    m_image.transition(cb, ImageState::TransferDst);

    std::vector<vk::BufferImageCopy> regions;
    regions.reserve(dds.mipLevelCount);
    WHEELS_ASSERT(dds.levelByteOffsets.size() == dds.mipLevelCount);
    for (uint32_t i = 0; i < dds.mipLevelCount; ++i)
    {
        regions.push_back(vk::BufferImageCopy{
            .bufferOffset =
                asserted_cast<vk::DeviceSize>(dds.levelByteOffsets[i]),
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource =
                vk::ImageSubresourceLayers{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = i,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            .imageOffset = {0, 0, 0},
            .imageExtent =
                {
                    std::max(extent.width >> i, 1u),
                    std::max(extent.height >> i, 1u),
                    1u,
                },
        });
    }

    cb.copyBufferToImage(
        stagingBuffer.handle, m_image.handle,
        vk::ImageLayout::eTransferDstOptimal,
        asserted_cast<uint32_t>(regions.size()), regions.data());

    if (options.initialState != ImageState::Unknown)
        m_image.transition(cb, options.initialState);
}

vk::DescriptorImageInfo Texture2D::imageInfo() const
{
    return vk::DescriptorImageInfo{
        .imageView = m_image.view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
}

void Texture3D::init(
    ScopedScratch scopeAlloc, const std::filesystem::path &path,
    const ImageState initialState)
{
    Dds dds = readDds(scopeAlloc, path);
    WHEELS_ASSERT(!dds.data.empty());

    const vk::Extent3D extent{
        dds.width,
        dds.height,
        dds.depth,
    };

    // Just create the staging here as Texture3D are only loaded in during load
    // time so we can wait for upload to complete
    Buffer stagingBuffer = gDevice.createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = dds.data.size(),
                .usage = vk::BufferUsageFlagBits::eTransferSrc,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .debugName = "Texture3DStaging",
    });
    defer { gDevice.destroy(stagingBuffer); };

    WHEELS_ASSERT(dds.data.size() <= stagingBuffer.byteSize);
    memcpy(stagingBuffer.mapped, dds.data.data(), dds.data.size());

    const std::filesystem::path relPath = relativePath(path);

    WHEELS_ASSERT(dds.mipLevelCount == 1);
    m_image = gDevice.createImage(ImageCreateInfo{
        .desc =
            ImageDescription{
                .imageType = vk::ImageType::e3D,
                .format = asVkFormat(dds.format),
                .width = extent.width,
                .height = extent.height,
                .depth = extent.depth,
                .layerCount = 1,
                .usageFlags = vk::ImageUsageFlagBits::eTransferSrc |
                              vk::ImageUsageFlagBits::eTransferDst |
                              vk::ImageUsageFlagBits::eSampled,
            },
        .debugName = relPath.generic_string().c_str(),
    });

    // Just create an ad hoc cb here as Texture3D are only loaded in during load
    // time so we can wait for upload to complete
    const vk::CommandBuffer cb = gDevice.beginGraphicsCommands();

    m_image.transition(cb, ImageState::TransferDst);

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
        .imageExtent = extent,
    };

    cb.copyBufferToImage(
        stagingBuffer.handle, m_image.handle,
        vk::ImageLayout::eTransferDstOptimal, 1, &region);

    if (initialState != ImageState::Unknown)
        m_image.transition(cb, initialState);

    gDevice.endGraphicsCommands(cb);
}

vk::DescriptorImageInfo Texture3D::imageInfo() const
{
    return vk::DescriptorImageInfo{
        .imageView = m_image.view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
}

void TextureCubemap::init(
    ScopedScratch scopeAlloc, const std::filesystem::path &path)
{
    const Ktx cube = readKtx(scopeAlloc, path);
    WHEELS_ASSERT(cube.faceCount == 6);

    WHEELS_ASSERT(
        cube.width == 512 && cube.height == 512 &&
        "Diffuse irradiance gather expects input as 512x512 to sample from the "
        "correct mip");
    WHEELS_ASSERT(
        cube.mipLevelCount > 4 &&
        "Diffuse irradiance gather happens from mip 3");

    const std::filesystem::path relPath = relativePath(path);

    m_image = gDevice.createImage(ImageCreateInfo{
        .desc =
            ImageDescription{
                .format = cube.format,
                .width = cube.width,
                .height = cube.height,
                .mipCount = cube.mipLevelCount,
                .layerCount = cube.faceCount,
                .createFlags = vk::ImageCreateFlagBits::eCubeCompatible,
                .usageFlags = vk::ImageUsageFlagBits::eTransferDst |
                              vk::ImageUsageFlagBits::eSampled,
            },
        .debugName = relPath.generic_string().c_str(),
    });

    copyPixels(scopeAlloc.child_scope(), cube, m_image.subresourceRange);

    m_sampler = gDevice.logical().createSampler(vk::SamplerCreateInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .anisotropyEnable = VK_TRUE,
        .maxAnisotropy = 16,
        .minLod = 0,
        .maxLod = static_cast<float>(cube.mipLevelCount),
    });
}

TextureCubemap::~TextureCubemap() { gDevice.logical().destroy(m_sampler); }

TextureCubemap::TextureCubemap(TextureCubemap &&other) noexcept
: Texture{WHEELS_MOV(other)}
, m_sampler{other.m_sampler}
{
}

TextureCubemap &TextureCubemap::operator=(TextureCubemap &&other) noexcept
{
    if (this != &other)
    {
        destroy();

        m_image = WHEELS_MOV(other.m_image);
        m_sampler = other.m_sampler;

        other.m_image = Image{};
        other.m_sampler = vk::Sampler{};
    }
    return *this;
}

vk::DescriptorImageInfo TextureCubemap::imageInfo() const
{
    return vk::DescriptorImageInfo{
        .sampler = m_sampler,
        .imageView = m_image.view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
}

void TextureCubemap::copyPixels(
    ScopedScratch scopeAlloc, const Ktx &cube,
    const vk::ImageSubresourceRange &subresourceRange) const
{
    Buffer stagingBuffer = gDevice.createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = cube.data.size(),
                .usage = vk::BufferUsageFlagBits::eTransferSrc,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .debugName = "TextureCubemapStaging",
    });
    defer { gDevice.destroy(stagingBuffer); };

    memcpy(stagingBuffer.mapped, cube.data.data(), cube.data.size());

    // Collect memory regions of all faces and their miplevels so their
    // transfers can be submitted together
    const Array<vk::BufferImageCopy> regions = [&]
    {
        WHEELS_ASSERT(cube.arrayLayerCount == 1);

        Array<vk::BufferImageCopy> regions{
            scopeAlloc, asserted_cast<size_t>(cube.faceCount) *
                            asserted_cast<size_t>(cube.mipLevelCount)};
        for (uint32_t iMip = 0; iMip < cube.mipLevelCount; ++iMip)
        {
            for (uint32_t iFace = 0; iFace < cube.faceCount; ++iFace)
            {
                const uint32_t width = std::max(cube.width >> iMip, 1u);
                const uint32_t height = std::max(cube.height >> iMip, 1u);
                const uint32_t sourceOffset =
                    cube.levelByteOffsets[iMip * cube.faceCount + iFace];

                regions.push_back(vk::BufferImageCopy{
                    .bufferOffset = sourceOffset,
                    .bufferRowLength = 0,
                    .bufferImageHeight = 0,
                    .imageSubresource =
                        vk::ImageSubresourceLayers{
                            .aspectMask = vk::ImageAspectFlagBits::eColor,
                            .mipLevel = iMip,
                            .baseArrayLayer = iFace,
                            .layerCount = 1,
                        },
                    .imageOffset = vk::Offset3D{0},
                    .imageExtent = vk::Extent3D{width, height, 1},
                });
            }
        }
        return regions;
    }();

    const auto copyBuffer = gDevice.beginGraphicsCommands();

    transitionImageLayout(
        copyBuffer, m_image.handle, subresourceRange,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
        vk::AccessFlags{}, vk::AccessFlagBits::eTransferWrite,
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer);

    copyBuffer.copyBufferToImage(
        stagingBuffer.handle, m_image.handle,
        vk::ImageLayout::eTransferDstOptimal,
        asserted_cast<uint32_t>(regions.size()), regions.data());

    transitionImageLayout(
        copyBuffer, m_image.handle, subresourceRange,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader);

    gDevice.endGraphicsCommands(copyBuffer);
}
