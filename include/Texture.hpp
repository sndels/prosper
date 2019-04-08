#ifndef PROSPER_TEXTURE_HPP
#define PROSPER_TEXTURE_HPP

#include <tiny_gltf.h>

#include "Device.hpp"

class Texture {
public:
    Texture(Device* device, const std::string& path);
    Texture(Device* device, const tinygltf::Image& image, const tinygltf::Sampler& sampler);
    ~Texture();

    vk::DescriptorImageInfo imageInfo() const;

private:
    std::pair<uint8_t*, vk::Extent2D> loadFromFile(const std::string& path);
    Buffer stagePixels(const uint8_t* pixels, const vk::Extent2D extent);
    void createImage(const Buffer& stagingBuffer, const vk::Extent2D extent, const vk::ImageSubresourceRange& subresourceRange);
    void createImageView(const vk::ImageSubresourceRange& subresourceRange);
    void createSampler();
    void createSampler(const tinygltf::Sampler& sampler);

    Device* _device;
    Image _image;
    vk::ImageView _imageView;
    vk::Sampler _sampler;

};

#endif // PROSPER_TEXTURE_HPP
