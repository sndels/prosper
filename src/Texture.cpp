#include "Texture.hpp"

#include <iostream>

#include <stb_image.h>

namespace {
    vk::Filter getVkFilterMode(int glEnum)
    {
        switch (glEnum) {
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
        switch (glEnum) {
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

    std::pair<uint8_t*, vk::Extent2D> pixelsFromFile(const std::string& path)
    {
        int w, h, channels;
        stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &channels, STBI_rgb_alpha);
        if (pixels == nullptr)
            throw std::runtime_error("Failed to load texture '" + path + "'");

        return std::pair{
            pixels,
            vk::Extent2D{
                static_cast<uint32_t>(w),
                static_cast<uint32_t>(h)
            }
        };
    }
}

Texture::Texture(Device* device, const std::string& path) :
    _device(device)
{
    const auto [pixels, extent] = pixelsFromFile(path);
    const auto stagingBuffer = stagePixels(pixels, extent);

    const vk::ImageSubresourceRange subresourceRange{
            vk::ImageAspectFlagBits::eColor,
            0, // baseMipLevel
            1, // levelCount
            0, // baseArrayLayer
            1 // layerCount
    };

    createImage(stagingBuffer, extent, subresourceRange);
    createImageView(subresourceRange);
    createSampler();

    stbi_image_free(pixels);
    _device->logical().destroy(stagingBuffer.handle);
    _device->logical().free(stagingBuffer.memory);
}

Texture::Texture(Device* device, const tinygltf::Image& image, const tinygltf::Sampler& sampler) :
    _device(device)
{
    // TODO: support
    if (image.pixel_type != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
        throw std::runtime_error("Unsupported glTF pixel_type");

    const uint8_t* pixels;
    std::vector<uint8_t> tmpPixels;
    if (image.component < 3)
        throw std::runtime_error("Image with less than 3 components");
    else if (image.component == 3) {
        std::cerr << "3 component texture" << std::endl;
        // Add fourth channel
        // TODO: Do only if rgb-textures are unsupported
        tmpPixels.resize(image.width * image.height * 4);
        const auto* rgb = image.image.data();
        auto* rgba = tmpPixels.data();
        for (size_t i = 0; i < image.width * image.height; ++i) {
            rgba[0] = rgb[0];
            rgba[1] = rgb[1];
            rgba[2] = rgb[2];
            rgb += 3;
            rgba += 4;
        }
        pixels = tmpPixels.data();
    } else
        pixels = reinterpret_cast<const uint8_t*>(image.image.data());
    const VkExtent2D extent{
        static_cast<uint32_t>(image.width),
        static_cast<uint32_t>(image.height)
    };
    const auto stagingBuffer = stagePixels(pixels, extent);

    const vk::ImageSubresourceRange subresourceRange{
            vk::ImageAspectFlagBits::eColor,
            0, // baseMipLevel
            1, // levelCount
            0, // baseArrayLayer
            1 // layerCount
    };

    createImage(stagingBuffer, extent, subresourceRange);
    createImageView(subresourceRange);
    createSampler(sampler);

    _device->logical().destroy(stagingBuffer.handle);
    _device->logical().free(stagingBuffer.memory);
}

Texture::~Texture()
{
    _device->logical().destroy(_imageView);
    _device->logical().destroy(_sampler);
    _device->logical().destroy(_image.handle);
    _device->logical().free(_image.memory);
}

vk::DescriptorImageInfo Texture::imageInfo() const
{
    return vk::DescriptorImageInfo{
        _sampler,
        _imageView,
        vk::ImageLayout::eShaderReadOnlyOptimal
    };
}

Buffer Texture::stagePixels(const uint8_t* pixels, const vk::Extent2D extent)
{
    const vk::DeviceSize imageSize = extent.width * extent.height * 4;

    const Buffer stagingBuffer = _device->createBuffer(
        imageSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent
    );

    void* data;
    _device->logical().mapMemory(stagingBuffer.memory, 0, imageSize, {}, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    _device->logical().unmapMemory(stagingBuffer.memory);

    return stagingBuffer;
}

void Texture::createImage(const Buffer& stagingBuffer, const vk::Extent2D extent, const vk::ImageSubresourceRange& subresourceRange)
{
    _image = _device->createImage(
        extent,
        vk::Format::eR8G8B8A8Unorm,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst |
        vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    _device->transitionImageLayout(
        _image,
        subresourceRange,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal
    );
    _device->copyBufferToImage(stagingBuffer, _image, extent);
    _device->transitionImageLayout(
        _image,
        subresourceRange,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal
    );
}

void Texture::createImageView(const vk::ImageSubresourceRange& subresourceRange)
{
    _imageView = _device->logical().createImageView({
        {}, // flags
        _image.handle,
        vk::ImageViewType::e2D,
        vk::Format::eR8G8B8A8Unorm,
        vk::ComponentMapping{},
        subresourceRange
    });
}

void Texture::createSampler()
{
    // TODO: Use shared samplers
    _sampler = _device->logical().createSampler({
        {}, // flags
        vk::Filter::eLinear, // magFilter
        vk::Filter::eLinear, // minFilter
        vk::SamplerMipmapMode::eLinear,
        vk::SamplerAddressMode::eClampToEdge, // addressModeU
        vk::SamplerAddressMode::eClampToEdge, // addressModeV
        vk::SamplerAddressMode::eClampToEdge, // addressModeW
        0.f, // mipLodBias
        VK_TRUE, // anisotropyEnable
        16 // maxAnisotropy
    });
}

void Texture::createSampler(const tinygltf::Sampler& sampler)
{
    // TODO: Use shared samplers
    _sampler = _device->logical().createSampler({
        {}, // flags
        getVkFilterMode(sampler.magFilter),
        getVkFilterMode(sampler.minFilter),
        vk::SamplerMipmapMode::eLinear, // TODO
        getVkAddressMode(sampler.wrapS),
        getVkAddressMode(sampler.wrapT),
        vk::SamplerAddressMode::eClampToEdge, // addressModeW
        0.f, // mipLodBias
        VK_TRUE, // anisotropyEnable
        16 // maxAnisotropy
    });
}
