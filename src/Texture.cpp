#include "Texture.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <variant>

#include <stb_image.h>
#include <tiny_gltf.h>
#include <wheels/containers/array.hpp>
#include <wheels/containers/pair.hpp>

#include "Utils.hpp"

using namespace wheels;

namespace
{

// This should be incremented when breaking changes are made to what's cached
const uint32_t sShaderCacheVersion = 0;

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
        uint32_t cacheVersion;
        assert(sizeof(cacheVersion) == sizeof(sShaderCacheVersion));
        // NOTE:
        // Caches aren't supposed to be portable so this doesn't pay attention
        // to endianness.
        tagFile.read(
            reinterpret_cast<char *>(&cacheVersion), sizeof(cacheVersion));

        if (sShaderCacheVersion != cacheVersion)
            return false;
    }
    catch (std::exception)
    {
        return false;
    }

    return true;
}

void compress(
    const std::filesystem::path &targetPath,
    const UncompressedPixelData &pixels)
{
    gli::texture2d tex{
        gli::FORMAT_RGBA8_UNORM_PACK8,
        glm::ivec2{pixels.extent.width, pixels.extent.height}, 1};
    memcpy(
        tex.data(), pixels.data,
        pixels.extent.width * pixels.extent.height * pixels.channels);

    // TODO: Mips
    // TODO: Compression

    assert(gli::save(tex, targetPath.string()) && "GLI save failed");

    const std::filesystem::path tagPath = cacheTagPath(targetPath);
    std::ofstream tagFile{tagPath, std::ios_base::binary};
    // NOTE:
    // Caches aren't supposed to be portable so this doesn't pay attention to
    // endianness.
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

void stagePixels(
    const Buffer &stagingBuffer, const void *data, vk::Extent2D extent,
    uint32_t bytesPerPixel)
{
    assert(stagingBuffer.mapped != nullptr);
    assert(data != nullptr);

    const vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(extent.width) *
                                     extent.height * bytesPerPixel;
    assert(imageSize <= stagingBuffer.byteSize);

    memcpy(stagingBuffer.mapped, data, asserted_cast<size_t>(imageSize));
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

Texture2D::Texture2D(
    ScopedScratch scopeAlloc, Device *device, const std::filesystem::path &path,
    vk::CommandBuffer cb, const Buffer &stagingBuffer, const bool mipmap)
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

        compress(cached, pixels);

        stbi_image_free(pixels.dataOwned);
    }

    const gli::texture2d tex(gli::load(cached.string()));
    assert(!tex.empty());

    assert(tex.format() == gli::FORMAT_RGBA8_UNORM_PACK8);
    const uint32_t bytesPerPixel = 4;

    const vk::Extent2D extent{

        asserted_cast<uint32_t>(tex.extent().x),
        asserted_cast<uint32_t>(tex.extent().y),
    };

    stagePixels(stagingBuffer, tex.data(), extent, bytesPerPixel);

    const uint32_t mipLevels =
        mipmap ? asserted_cast<uint32_t>(
                     floor(log2(std::max(extent.width, extent.height)))) +
                     1
               : 1;

    createImage(
        cb, stagingBuffer,
        ImageCreateInfo{
            .desc =
                ImageDescription{
                    .format = vk::Format::eR8G8B8A8Unorm,
                    .width = extent.width,
                    .height = extent.height,
                    .mipCount = mipLevels,
                    .layerCount = 1,
                    .usageFlags = vk::ImageUsageFlagBits::eTransferSrc |
                                  vk::ImageUsageFlagBits::eTransferDst |
                                  vk::ImageUsageFlagBits::eSampled,
                },
            .debugName = "Texture2D",
        });
}

Texture2D::Texture2D(
    ScopedScratch scopeAlloc, Device *device, const tinygltf::Image &image,
    vk::CommandBuffer cb, const Buffer &stagingBuffer, const bool mipmap)
: Texture(device)
{
    assert(device != nullptr);

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

    stagePixels(stagingBuffer, pixels.data, pixels.extent, pixels.channels);

    const uint32_t mipLevels =
        mipmap ? asserted_cast<uint32_t>(floor(log2(
                     std::max(pixels.extent.width, pixels.extent.height)))) +
                     1
               : 1;

    createImage(
        cb, stagingBuffer,
        ImageCreateInfo{
            .desc =
                ImageDescription{
                    .format = vk::Format::eR8G8B8A8Unorm,
                    .width = pixels.extent.width,
                    .height = pixels.extent.height,
                    .mipCount = mipLevels,
                    .layerCount = 1,
                    .usageFlags = vk::ImageUsageFlagBits::eTransferSrc |
                                  vk::ImageUsageFlagBits::eTransferDst |
                                  vk::ImageUsageFlagBits::eSampled,
                },
            .debugName = "Texture2D",
        });
}

vk::DescriptorImageInfo Texture2D::imageInfo() const
{
    return vk::DescriptorImageInfo{
        .imageView = _image.view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
}

void Texture2D::createImage(
    vk::CommandBuffer cb, const Buffer &stagingBuffer,
    const ImageCreateInfo &info)
{
    // Both transfer source and destination as pixels will be transferred to it
    // and mipmaps will be generated from it
    _image = _device->createImage(info);

    _image.transition(
        cb, ImageState{
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
    cb.copyBufferToImage(
        stagingBuffer.handle, _image.handle,
        vk::ImageLayout::eTransferDstOptimal, 1, &region);

    createMipmaps(
        cb,
        vk::Extent2D{
            .width = info.desc.width,
            .height = info.desc.height,
        },
        info.desc.mipCount);
}

void Texture2D::createMipmaps(
    vk::CommandBuffer cb, const vk::Extent2D &extent, const uint32_t mipLevels)
{
    // TODO: Check that the texture format supports linear filtering
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
            cb, _image.handle, subresourceRange,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eTransferSrcOptimal,
            vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eTransferRead,
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eTransfer);

        const vk::ImageBlit blit{
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
        cb.blitImage(
            _image.handle, vk::ImageLayout::eTransferSrcOptimal, _image.handle,
            vk::ImageLayout::eTransferDstOptimal, 1, &blit,
            vk::Filter::eLinear);

        // Source needs to be transitioned to shader read optimal
        transitionImageLayout(
            cb, _image.handle, subresourceRange,
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
        cb, _image.handle, subresourceRange,
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
}

TextureCubemap::TextureCubemap(
    ScopedScratch scopeAlloc, Device *device, const std::filesystem::path &path)
: Texture(device)
{
    assert(device != nullptr);

    const gli::texture_cube cube(gli::load(path.string()));
    assert(!cube.empty());
    assert(cube.faces() == 6);

    const auto mipLevels = asserted_cast<uint32_t>(cube.levels());

    _image = _device->createImage(ImageCreateInfo{
        .desc =
            ImageDescription{
                .format = vk::Format::eR16G16B16A16Sfloat,
                .width = asserted_cast<uint32_t>(cube.extent().x),
                .height = asserted_cast<uint32_t>(cube.extent().y),
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
