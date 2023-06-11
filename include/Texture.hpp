#ifndef PROSPER_TEXTURE_HPP
#define PROSPER_TEXTURE_HPP

#include "Device.hpp"

#include <wheels/allocators/scoped_scratch.hpp>

#include <filesystem>

namespace gli
{
class texture_cube;
}

namespace tinygltf
{
struct Image;
}; // namespace tinygltf

class Texture
{
  public:
    Texture(Device *device);
    virtual ~Texture();

    Texture(const Texture &other) = delete;
    Texture(Texture &&other) noexcept;
    Texture &operator=(const Texture &other) = delete;
    Texture &operator=(Texture &&other) noexcept;

    [[nodiscard]] virtual vk::DescriptorImageInfo imageInfo() const = 0;

  protected:
    void destroy();

    // Texture with null device is invalid or moved
    Device *_device{nullptr};
    Image _image;
};

class Texture2D : public Texture
{
  public:
    // The image is ready and stagingBuffer can be freed once cb is submitted
    // and has finished executing.
    Texture2D(
        wheels::ScopedScratch scopeAlloc, Device *device,
        const std::filesystem::path &path, vk::CommandBuffer cb,
        const Buffer &stagingBuffer, bool mipmap);

    [[nodiscard]] vk::DescriptorImageInfo imageInfo() const override;

  private:
    void createImage(
        vk::CommandBuffer cb, const Buffer &stagingBuffer,
        const ImageCreateInfo &info);
    void createMipmaps(
        vk::CommandBuffer cb, const vk::Extent2D &extent, uint32_t mipLevels);
};

class TextureCubemap : public Texture
{
  public:
    TextureCubemap(
        wheels::ScopedScratch scopeAlloc, Device *device,
        const std::filesystem::path &path);
    ~TextureCubemap() override;

    TextureCubemap(const TextureCubemap &other) = delete;
    TextureCubemap(TextureCubemap &&other) noexcept;
    TextureCubemap &operator=(const TextureCubemap &other) = delete;
    TextureCubemap &operator=(TextureCubemap &&other) noexcept;

    [[nodiscard]] vk::DescriptorImageInfo imageInfo() const override;

  private:
    void copyPixels(
        wheels::ScopedScratch scopeAlloc, const gli::texture_cube &cube,
        const vk::ImageSubresourceRange &subresourceRange) const;

    vk::Sampler _sampler;
};

#endif // PROSPER_TEXTURE_HPP
