#include "Texture.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <variant>

// GLI uses rgba accesses
#undef GLM_FORCE_XYZW_ONLY
#include <gli/gli.hpp>
#define GLM_FORCE_XYZW_ONLY

#include <ispc_texcomp.h>
#include <stb_image.h>
#include <tiny_gltf.h>
#include <wheels/containers/array.hpp>
#include <wheels/containers/pair.hpp>

#include "../gfx/Device.hpp"
#include "../utils/Dds.hpp"
#include "../utils/Utils.hpp"

using namespace wheels;

namespace
{

// This should be incremented when breaking changes are made to what's cached
const uint32_t sShaderCacheVersion = 2;

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

    static_assert(sizeof(uint8_t) == sizeof(stbi_uc));
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

// Returns path of the corresponding cached file, creating the folders up to
// it if those don't exist
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

bool cacheValid(const std::filesystem::path &cacheFile)
{
    try
    {
        if (!std::filesystem::exists(cacheFile))
            return false;

        const std::filesystem::path tagPath = cacheTagPath(cacheFile);
        if (!std::filesystem::exists(tagPath))
            return false;

        std::ifstream tagFile{tagPath};
        uint32_t cacheVersion = 0xFFFFFFFFu;
        static_assert(sizeof(cacheVersion) == sizeof(sShaderCacheVersion));
        // NOTE:
        // Caches aren't supposed to be portable so this doesn't pay attention
        // to endianness.
        tagFile.read(
            reinterpret_cast<char *>(&cacheVersion), sizeof(cacheVersion));

        if (sShaderCacheVersion != cacheVersion)
            return false;
    }
    catch (std::exception &)
    {
        return false;
    }

    return true;
}

void generateMipLevels(
    Array<uint8_t> &rawLevels, Array<uint32_t> &rawLevelByteOffsets,
    const UncompressedPixelData &pixels)
{
    // GLI's implementation generated weird artifacts on sponza's fabrics so
    // let's do this ourselves.
    // TODO:
    // - Optimize
    // - Better algo for e.g. normals?
    const size_t mipLevelCount = rawLevelByteOffsets.size();
    // TODO: Non-8bit channels?
    const uint32_t pixelStride = pixels.channels;
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
        for (uint32_t j = 0; j < height; ++j)
        {
            // Clamp to edge
            const uint32_t y0ParentOffset =
                std::min(j * 2, parentHeight - 1) * parentRowStride;
            const uint32_t y1ParentOffset =
                std::min(j * 2 + 1, parentHeight - 1) * parentRowStride;
            for (uint32_t i = 0; i < width; ++i)
            {
                // Clamp to edge
                const uint32_t x0ParentOffset =
                    std::min(i * 2, parentWidth - 1) * pixelStride;
                const uint32_t x1ParentOffset =
                    std::min(i * 2 + 1, parentWidth - 1) * pixelStride;
                for (uint32_t c = 0; c < pixels.channels; ++c)
                {
                    const uint16_t v00 =
                        parentData[y0ParentOffset + x0ParentOffset + c];
                    const uint16_t v01 =
                        parentData[y0ParentOffset + x1ParentOffset + c];
                    const uint16_t v10 =
                        parentData[y1ParentOffset + x0ParentOffset + c];
                    const uint16_t v11 =
                        parentData[y1ParentOffset + x1ParentOffset + c];

                    // Linear filter
                    data[(j * width + i) * pixelStride + c] =
                        static_cast<uint8_t>(
                            static_cast<float>(v00 + v01 + v10 + v11) / 4.f);
                }
            }
        }
    }
}

void compress(
    ScopedScratch scopeAlloc, const std::filesystem::path &targetPath,
    const UncompressedPixelData &pixels, bool generateMips)
{
    // First calculate mip count down to 1x1
    const int32_t fullMipLevelCount =
        generateMips
            ? asserted_cast<int32_t>(floor(
                  log2(std::max(pixels.extent.width, pixels.extent.height)))) +
                  1
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

    Dds dds{
        scopeAlloc, pixels.extent.width, pixels.extent.height, format,
        mipLevelCount};

    if (format == DxgiFormat::BC7Unorm)
    {
        Array<uint8_t> rawLevels{scopeAlloc};
        // Twice the size of the first level should be plenty for mips
        const size_t inputSize = asserted_cast<size_t>(pixels.extent.width) *
                                 asserted_cast<size_t>(pixels.extent.height) *
                                 4;
        rawLevels.resize(inputSize * 2);

        memcpy(rawLevels.data(), pixels.data, inputSize);

        Array<uint32_t> rawLevelByteOffsets{scopeAlloc};
        rawLevelByteOffsets.resize(mipLevelCount);

        if (mipLevelCount > 1)
            generateMipLevels(rawLevels, rawLevelByteOffsets, pixels);
        else
            rawLevelByteOffsets[0] = 0u;

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
            CompressBlocksBC7(&rgbaSurface, dst, &bc7Settings);
        }
    }
    else
    {
        Array<uint8_t> rawLevels{scopeAlloc};
        const size_t inputSize = asserted_cast<size_t>(pixels.extent.width) *
                                 asserted_cast<size_t>(pixels.extent.height) *
                                 4;
        rawLevels.resize(inputSize * 2);

        memcpy(rawLevels.data(), pixels.data, inputSize);

        Array<uint32_t> rawLevelByteOffsets{scopeAlloc};
        rawLevelByteOffsets.resize(mipLevelCount);
        if (mipLevelCount > 1)
            generateMipLevels(rawLevels, rawLevelByteOffsets, pixels);

        WHEELS_ASSERT(dds.data.size() <= rawLevels.size());
        memcpy(dds.data.data(), rawLevels.data(), dds.data.size());
    }

    writeDds(dds, targetPath);

    const std::filesystem::path tagPath = cacheTagPath(targetPath);
    std::ofstream tagFile{tagPath, std::ios_base::binary};
    // NOTE:
    // Caches aren't supposed to be portable so this doesn't pay
    // attention to endianness.
    tagFile.write(
        reinterpret_cast<const char *>(&sShaderCacheVersion),
        sizeof(sShaderCacheVersion));
    tagFile.close();

    // Make sure we have rw permissions for the user to be nice
    const std::filesystem::perms initialPerms =
        std::filesystem::status(tagPath).permissions();
    std::filesystem::permissions(
        tagPath, initialPerms | std::filesystem::perms::owner_read |
                     std::filesystem::perms::owner_write);
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
    case DxgiFormat::BC7Unorm:
        return vk::Format::eBc7UnormBlock;
    default:
        break;
    }
    throw std::runtime_error("Unkown DxgiFormat");
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

vk::Image Texture::nativeHandle() const { return _image.handle; }

void Texture::destroy()
{
    if (_device != nullptr)
        _device->destroy(_image);
}

Texture2D::Texture2D(
    ScopedScratch scopeAlloc, Device *device, const std::filesystem::path &path,
    vk::CommandBuffer cb, const Buffer &stagingBuffer, const bool mipmap,
    const bool skipPostTransition)
: Texture(device)
{
    const auto cached = cachePath(path);
    if (!cacheValid(cached))
    {
        auto pixels = pixelsFromFile(path);

        Array<uint8_t> tmpPixels{scopeAlloc};
        if (pixels.channels < 3)
            throw std::runtime_error("Image with less than 3 components");

        if (pixels.channels == 3)
        {
            // Add fourth channel as 3 channel optimal tiling is rarely
            // supported
            tmpPixels.resize(
                asserted_cast<size_t>(pixels.extent.width) *
                pixels.extent.height * 4);
            const auto *rgb = pixels.data;
            auto *rgba = tmpPixels.data();
            for (auto i = 0u; i < pixels.extent.width * pixels.extent.height;
                 ++i)
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

        compress(scopeAlloc.child_scope(), cached, pixels, mipmap);

        stbi_image_free(pixels.dataOwned);
    }

    // TODO:
    // If cache was invalid, the newly cached one directly from memory
    Dds dds = readDds(scopeAlloc, cached);

    WHEELS_ASSERT(!dds.data.empty());

    const vk::Extent2D extent{
        asserted_cast<uint32_t>(dds.width),
        asserted_cast<uint32_t>(dds.height),
    };

    WHEELS_ASSERT(stagingBuffer.mapped != nullptr);
    WHEELS_ASSERT(dds.data.size() <= stagingBuffer.byteSize);

    memcpy(stagingBuffer.mapped, dds.data.data(), dds.data.size());

    // TODO:
    // Use srgb formats in dds for srgb data, have a flag in texture ctor for
    // using unorm for snorm inputs. Having srgb data say it's unorm is
    // potentially confusing.

    _image = _device->createImage(ImageCreateInfo{
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
        .debugName = "Texture2D",
    });

    _image.transition(cb, ImageState::TransferDst);

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
                    .mipLevel = asserted_cast<uint32_t>(i),
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
        stagingBuffer.handle, _image.handle,
        vk::ImageLayout::eTransferDstOptimal,
        asserted_cast<uint32_t>(regions.size()), regions.data());

    if (!skipPostTransition)
        _image.transition(
            cb, ImageState::FragmentShaderRead | ImageState::RayTracingRead);
}

vk::DescriptorImageInfo Texture2D::imageInfo() const
{
    return vk::DescriptorImageInfo{
        .imageView = _image.view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
}

TextureCubemap::TextureCubemap(
    ScopedScratch scopeAlloc, Device *device, const std::filesystem::path &path)
: Texture(device)
{
    WHEELS_ASSERT(device != nullptr);

    const gli::texture_cube cube(gli::load(path.string()));
    WHEELS_ASSERT(!cube.empty());
    WHEELS_ASSERT(cube.faces() == 6);

    const auto mipLevels = asserted_cast<uint32_t>(cube.levels());
    const gli::texture_cube::extent_type extent = cube.extent();
    WHEELS_ASSERT(
        extent.x == 512 && extent.y == 512 &&
        "Diffuse irradiance gather expects input as 512x512 to sample from the "
        "correct mip");
    WHEELS_ASSERT(
        mipLevels > 4 && "Diffuse irradiance gather happens from mip 3");

    _image = _device->createImage(ImageCreateInfo{
        .desc =
            ImageDescription{
                .format = vk::Format::eR16G16B16A16Sfloat,
                .width = asserted_cast<uint32_t>(extent.x),
                .height = asserted_cast<uint32_t>(extent.y),
                .mipCount = mipLevels,
                .layerCount = 6,
                .createFlags = vk::ImageCreateFlagBits::eCubeCompatible,
                .usageFlags = vk::ImageUsageFlagBits::eTransferDst |
                              vk::ImageUsageFlagBits::eSampled,
            },
        .debugName = "TextureCubemap",
    });

    copyPixels(scopeAlloc.child_scope(), cube, _image.subresourceRange);

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
: Texture{WHEELS_MOV(other)}
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
    ScopedScratch scopeAlloc, const gli::texture_cube &cube,
    const vk::ImageSubresourceRange &subresourceRange) const
{
    const Buffer stagingBuffer = _device->createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = cube.size(),
                .usage = vk::BufferUsageFlagBits::eTransferSrc,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .createMapped = true,
        .debugName = "TextureCubemapStaging",
    });

    memcpy(stagingBuffer.mapped, cube.data(), cube.size());

    // Collect memory regions of all faces and their miplevels so their
    // transfers can be submitted together
    const Array<vk::BufferImageCopy> regions = [&]
    {
        Array<vk::BufferImageCopy> regions{
            scopeAlloc, cube.faces() * cube.levels()};
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
