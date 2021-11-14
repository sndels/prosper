#include "Texture.hpp"

#include <cmath>
#include <iostream>

#include <stb_image.h>

#include "VkUtils.hpp"

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
        return vk::Filter::eNearest;
    case GL_LINEAR:
        return vk::Filter::eLinear;
    case GL_NEAREST_MIPMAP_NEAREST:
        return vk::Filter::eNearest;
    case GL_NEAREST_MIPMAP_LINEAR:
        return vk::Filter::eNearest;
    case GL_LINEAR_MIPMAP_NEAREST:
        return vk::Filter::eLinear;
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

std::pair<uint8_t *, vk::Extent2D> pixelsFromFile(const std::string &path)
{
    int w, h, channels;
    stbi_uc *pixels =
        stbi_load(path.c_str(), &w, &h, &channels, STBI_rgb_alpha);
    if (pixels == nullptr)
        throw std::runtime_error("Failed to load texture '" + path + "'");

    return std::pair{
        pixels,
        vk::Extent2D{static_cast<uint32_t>(w), static_cast<uint32_t>(h)}};
}
} // namespace

Texture::Texture(Device *device)
: _device{device}
{
}

Texture::~Texture() { destroy(); }

Texture::Texture(Texture &&other)
: _device{other._device}
, _image{other._image}
, _sampler{other._sampler}
{
    other._device = nullptr;
}

Texture &Texture::operator=(Texture &&other)
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
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
}

void Texture::destroy()
{
    if (_device)
    {
        _device->logical().destroy(_sampler);
        _device->destroy(_image);
    }
}

Texture2D::Texture2D(Device *device, const std::string &path, const bool mipmap)
: Texture(device)
{
    const auto [pixels, extent] = pixelsFromFile(path);
    const auto stagingBuffer = stagePixels(pixels, extent);

    const uint32_t mipLevels =
        mipmap ? static_cast<uint32_t>(
                     floor(log2(std::max(extent.width, extent.height)))) +
                     1
               : 1;
    const vk::ImageSubresourceRange subresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = mipLevels,
        .baseArrayLayer = 0,
        .layerCount = 1};

    createImage(stagingBuffer, extent, subresourceRange);
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

    const uint8_t *pixels;
    std::vector<uint8_t> tmpPixels;
    if (image.component < 3)
        throw std::runtime_error("Image with less than 3 components");
    else if (image.component == 3)
    {
        std::cerr << "3 component texture" << std::endl;
        // Add fourth channel
        // TODO: Do only if rgb-textures are unsupported
        tmpPixels.resize(image.width * image.height * 4);
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
        static_cast<uint32_t>(image.width),
        static_cast<uint32_t>(image.height)};
    const auto stagingBuffer = stagePixels(pixels, extent);

    const uint32_t mipLevels =
        mipmap ? static_cast<uint32_t>(
                     floor(log2(std::max(extent.width, extent.height)))) +
                     1
               : 1;
    const vk::ImageSubresourceRange subresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = mipLevels,
        .baseArrayLayer = 0,
        .layerCount = 1};

    createImage(stagingBuffer, extent, subresourceRange);
    createSampler(sampler, mipLevels);

    _device->destroy(stagingBuffer);
}

Buffer Texture2D::stagePixels(
    const uint8_t *pixels, const vk::Extent2D extent) const
{
    const vk::DeviceSize imageSize = extent.width * extent.height * 4;

    const Buffer stagingBuffer = _device->createBuffer(
        imageSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    void *data;
    _device->map(stagingBuffer.allocation, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    _device->unmap(stagingBuffer.allocation);

    return stagingBuffer;
}

void Texture2D::createImage(
    const Buffer &stagingBuffer, const vk::Extent2D extent,
    const vk::ImageSubresourceRange &subresourceRange)
{
    // Both transfer source and destination as pixels will be transferred to it
    // and mipmaps will be generated from it
    _image = _device->createImage(
        extent, vk::Format::eR8G8B8A8Unorm, subresourceRange,
        vk::ImageViewType::e2D, vk::ImageTiling::eOptimal,
        vk::ImageCreateFlags{},
        vk::ImageUsageFlagBits::eTransferSrc |
            vk::ImageUsageFlagBits::eTransferDst |
            vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, VMA_MEMORY_USAGE_GPU_ONLY);

    const auto commandBuffer = _device->beginGraphicsCommands();

    transitionImageLayout(
        commandBuffer, _image.handle, subresourceRange,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
        vk::AccessFlags{}, vk::AccessFlagBits::eTransferWrite,
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer);

    const vk::BufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
            vk::ImageSubresourceLayers{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {extent.width, extent.height, 1}};
    commandBuffer.copyBufferToImage(
        stagingBuffer.handle, _image.handle,
        vk::ImageLayout::eTransferDstOptimal, 1, &region);

    _device->endGraphicsCommands(commandBuffer);

    createMipmaps(extent, subresourceRange.levelCount);
}

void Texture2D::createMipmaps(
    const vk::Extent2D extent, const uint32_t mipLevels) const
{
    // TODO: Check that the texture format supports linear filtering
    const auto buffer = _device->beginGraphicsCommands();

    vk::ImageSubresourceRange subresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1};

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
                    .layerCount = 1},
            .srcOffsets =
                {{vk::Offset3D{0}, vk::Offset3D{mipWidth, mipHeight, 1}}},
            .dstSubresource =
                vk::ImageSubresourceLayers{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = i,
                    .baseArrayLayer = 0,
                    .layerCount = 1},
            .dstOffsets = {
                {vk::Offset3D{0}, vk::Offset3D{
                                      mipWidth > 1 ? mipWidth / 2 : 1,
                                      mipHeight > 1 ? mipHeight / 2 : 1, 1}}}};
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
        vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderRead,
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader);

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
        .maxLod = static_cast<float>(mipLevels)});
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
        .maxLod = static_cast<float>(mipLevels)});
}

TextureCubemap::TextureCubemap(Device *device, const std::string &path)
: Texture(device)
{
    const gli::texture_cube cube(gli::load(path));
    assert(!cube.empty());
    assert(cube.faces() == 6);

    const vk::Extent2D layerExtent{
        static_cast<uint32_t>(cube.extent().x),
        static_cast<uint32_t>(cube.extent().y)};
    const uint32_t mipLevels = static_cast<uint32_t>(cube.levels());

    const vk::ImageSubresourceRange subresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = mipLevels,
        .baseArrayLayer = 0,
        .layerCount = 6};

    _image = _device->createImage(
        layerExtent, vk::Format::eR16G16B16A16Sfloat, subresourceRange,
        vk::ImageViewType::eCube, vk::ImageTiling::eOptimal,
        vk::ImageCreateFlagBits::eCubeCompatible,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, VMA_MEMORY_USAGE_GPU_ONLY);

    copyPixels(cube, subresourceRange);

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
        .maxLod = static_cast<float>(mipLevels)});
}

void TextureCubemap::copyPixels(
    const gli::texture_cube &cube,
    const vk::ImageSubresourceRange &subresourceRange) const
{
    const Buffer stagingBuffer = _device->createBuffer(
        cube.size(), vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    void *data;
    _device->map(stagingBuffer.allocation, &data);
    memcpy(data, cube.data(), cube.size());
    _device->unmap(stagingBuffer.allocation);

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
                regions.emplace_back(
                    offset, 0,
                    0, // bufferRowLength, bufferImageHeight, it is tightly
                       // packed
                    vk::ImageSubresourceLayers{
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .mipLevel = mipLevel,
                        .baseArrayLayer = face,
                        .layerCount = 1,
                    },
                    vk::Offset3D{0},
                    vk::Extent3D{
                        static_cast<uint32_t>(cube[face][mipLevel].extent().x),
                        static_cast<uint32_t>(cube[face][mipLevel].extent().y),
                        1});
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
        static_cast<uint32_t>(regions.size()), regions.data());

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
