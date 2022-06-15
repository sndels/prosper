#include "Texture.hpp"

#include <cmath>
#include <iostream>

#include <stb_image.h>
#include <tiny_gltf.h>

#include "Utils.hpp"

#ifdef _WIN32
// Windows' header doesn't include these
#define GL_CLAMP_TO_EDGE 0x812F
#define GL_MIRRORED_REPEAT 0x8370
#endif // _WIN32 or _WIN64

namespace
{
vk::Filter getVkFilterMode(int glEnum)
{
    switch (glEnum)
    {
    case GL_NEAREST:
    case GL_NEAREST_MIPMAP_NEAREST:
    case GL_NEAREST_MIPMAP_LINEAR:
        return vk::Filter::eNearest;
    case GL_LINEAR:
    case GL_LINEAR_MIPMAP_NEAREST:
    case GL_LINEAR_MIPMAP_LINEAR:
        return vk::Filter::eLinear;
    }

    std::cerr << "Invalid gl filter " << glEnum << std::endl;
    return vk::Filter::eLinear;
}

vk::SamplerAddressMode getVkAddressMode(int glEnum)
{
    switch (glEnum)
    {
    case GL_CLAMP_TO_EDGE:
        return vk::SamplerAddressMode::eClampToEdge;
    case GL_MIRRORED_REPEAT:
        return vk::SamplerAddressMode::eMirroredRepeat;
    case GL_REPEAT:
        return vk::SamplerAddressMode::eRepeat;
    }
    std::cerr << "Invalid gl wrapping mode " << glEnum << std::endl;
    return vk::SamplerAddressMode::eClampToEdge;
}

std::pair<uint8_t *, vk::Extent2D> pixelsFromFile(
    const std::filesystem::path &path)
{
    const auto pathString = path.string();
    int w = 0;
    int h = 0;
    int channels = 0;
    stbi_uc *pixels =
        stbi_load(pathString.c_str(), &w, &h, &channels, STBI_rgb_alpha);
    if (pixels == nullptr)
        throw std::runtime_error("Failed to load texture '" + pathString + "'");

    return std::pair{
        pixels,
        vk::Extent2D{asserted_cast<uint32_t>(w), asserted_cast<uint32_t>(h)}};
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

} // namespace

Texture::Texture(Device *device)
: _device{device}
{
}

Texture::~Texture() { destroy(); }

Texture::Texture(Texture &&other) noexcept
: _device{other._device}
, _image{other._image}
, _sampler{other._sampler}
{
    other._device = nullptr;
}

Texture &Texture::operator=(Texture &&other) noexcept
{
    destroy();
    if (this != &other)
    {
        _device = other._device;
        _image = other._image;
        _sampler = other._sampler;

        other._device = nullptr;
    }
    return *this;
}

vk::DescriptorImageInfo Texture::imageInfo() const
{
    return vk::DescriptorImageInfo{
        .sampler = _sampler,
        .imageView = _image.view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
}

void Texture::destroy()
{
    if (_device != nullptr)
    {
        _device->logical().destroy(_sampler);
        _device->destroy(_image);
    }
}

Texture2D::Texture2D(
    Device *device, const std::filesystem::path &path, const bool mipmap)
: Texture(device)
{
    const auto [pixels, extent] = pixelsFromFile(path);
    const auto stagingBuffer = stagePixels(pixels, extent);

    const uint32_t mipLevels =
        mipmap ? asserted_cast<uint32_t>(
                     floor(log2(std::max(extent.width, extent.height)))) +
                     1
               : 1;

    createImage(
        stagingBuffer, ImageCreateInfo{
                           .format = vk::Format::eR8G8B8A8Unorm,
                           .width = extent.width,
                           .height = extent.height,
                           .mipCount = mipLevels,
                           .layerCount = 1,
                           .usageFlags = vk::ImageUsageFlagBits::eTransferSrc |
                                         vk::ImageUsageFlagBits::eTransferDst |
                                         vk::ImageUsageFlagBits::eSampled,
                           .debugName = "Texture2D",
                       });
    createSampler(mipLevels);

    stbi_image_free(pixels);
    _device->destroy(stagingBuffer);
}

Texture2D::Texture2D(
    Device *device, const tinygltf::Image &image,
    const tinygltf::Sampler &sampler, const bool mipmap)
: Texture(device)
{
    // TODO: support
    if (image.pixel_type != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
        throw std::runtime_error("Unsupported glTF pixel_type");

    const uint8_t *pixels = nullptr;
    std::vector<uint8_t> tmpPixels;
    if (image.component < 3)
        throw std::runtime_error("Image with less than 3 components");

    if (image.component == 3)
    {
        std::cerr << "3 component texture" << std::endl;
        // Add fourth channel
        // TODO: Do only if rgb-textures are unsupported
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
        pixels = tmpPixels.data();
    }
    else
        pixels = reinterpret_cast<const uint8_t *>(image.image.data());
    const vk::Extent2D extent{
        asserted_cast<uint32_t>(image.width),
        asserted_cast<uint32_t>(image.height)};
    const auto stagingBuffer = stagePixels(pixels, extent);

    const uint32_t mipLevels =
        mipmap ? asserted_cast<uint32_t>(
                     floor(log2(std::max(extent.width, extent.height)))) +
                     1
               : 1;

    createImage(
        stagingBuffer, ImageCreateInfo{
                           .format = vk::Format::eR8G8B8A8Unorm,
                           .width = extent.width,
                           .height = extent.height,
                           .mipCount = mipLevels,
                           .layerCount = 1,
                           .usageFlags = vk::ImageUsageFlagBits::eTransferSrc |
                                         vk::ImageUsageFlagBits::eTransferDst |
                                         vk::ImageUsageFlagBits::eSampled,
                           .debugName = "Texture2D",
                       });
    createSampler(sampler, mipLevels);

    _device->destroy(stagingBuffer);
}

Buffer Texture2D::stagePixels(
    const uint8_t *pixels, const vk::Extent2D &extent) const
{
    const vk::DeviceSize imageSize =
        static_cast<vk::DeviceSize>(extent.width) * extent.height * 4;

    const Buffer stagingBuffer = _device->createBuffer(BufferCreateInfo{
        .byteSize = imageSize,
        .usage = vk::BufferUsageFlagBits::eTransferSrc,
        .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                      vk::MemoryPropertyFlagBits::eHostCoherent,
        .createMapped = true,
        .debugName = "Texture2DStaging",
    });

    memcpy(stagingBuffer.mapped, pixels, asserted_cast<size_t>(imageSize));

    return stagingBuffer;
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
            vk::PipelineStageFlagBits::eFragmentShader);

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
        vk::PipelineStageFlagBits::eFragmentShader);

    // We went around the state management since it doesn't support
    // per-subresource barriers
    _image.state = ImageState{
        .stageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .accessMask = vk::AccessFlagBits2::eShaderRead,
        .layout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    _device->endGraphicsCommands(buffer);
}

void Texture2D::createSampler(const uint32_t mipLevels)
{
    // TODO: Use shared samplers
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

void Texture2D::createSampler(
    const tinygltf::Sampler &sampler, const uint32_t mipLevels)
{
    // TODO: Use shared samplers
    _sampler = _device->logical().createSampler(vk::SamplerCreateInfo{
        .magFilter = getVkFilterMode(sampler.magFilter),
        .minFilter = getVkFilterMode(sampler.minFilter),
        .mipmapMode = vk::SamplerMipmapMode::eLinear, // TODO
        .addressModeU = getVkAddressMode(sampler.wrapS),
        .addressModeV = getVkAddressMode(sampler.wrapT),
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .anisotropyEnable = VK_TRUE,
        .maxAnisotropy = 16,
        .minLod = 0,
        .maxLod = static_cast<float>(mipLevels),
    });
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
