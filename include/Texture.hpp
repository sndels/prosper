#ifndef PROSPER_TEXTURE_HPP
#define PROSPER_TEXTURE_HPP
#include <stb_image.h>

#include "Device.hpp"

class Texture {
public:
    Texture(Device* device, const std::string& path);
    ~Texture();

    vk::DescriptorImageInfo imageInfo() const;

private:
    std::pair<uint8_t*, vk::Extent2D> loadFromFile(const std::string& path);
    Buffer stagePixels(const uint8_t* pixels, vk::Extent2D extent);
    void createImage(const Buffer& stagingBuffer, vk::Extent2D extent, const vk::ImageSubresourceRange& subresourceRange);
    void createImageView(const vk::ImageSubresourceRange& subresourceRange);
    void createSampler();

    Device* _device;
    Image _image;
    vk::ImageView _imageView;
    vk::Sampler _sampler;

};

#endif // PROSPER_TEXTURE_HPP
