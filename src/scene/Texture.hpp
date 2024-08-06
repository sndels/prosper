#ifndef PROSPER_SCENE_TEXTURE_HPP
#define PROSPER_SCENE_TEXTURE_HPP

#include "../gfx/Fwd.hpp"
#include "../gfx/Resources.hpp"
#include "../utils/Fwd.hpp"

#include <filesystem>
#include <wheels/allocators/scoped_scratch.hpp>

class Texture
{
  public:
    Texture() noexcept = default;
    virtual ~Texture();

    Texture(const Texture &other) = delete;
    Texture(Texture &&other) noexcept;
    Texture &operator=(const Texture &other) = delete;
    Texture &operator=(Texture &&other) noexcept;

    [[nodiscard]] virtual vk::DescriptorImageInfo imageInfo() const = 0;
    [[nodiscard]] virtual vk::Image nativeHandle() const;

  protected:
    void destroy();

    Image m_image;
};

enum class TextureColorSpace : uint8_t
{
    sRgb,
    Linear,
};

struct Texture2DOptions
{
    bool generateMipMaps{false};
    TextureColorSpace colorSpace{TextureColorSpace::sRgb};
    ImageState initialState{ImageState::Unknown};
};

class Texture2D : public Texture
{
  public:
    // The image is ready and stagingBuffer can be freed once cb is submitted
    // and has finished executing.
    void init(
        wheels::ScopedScratch scopeAlloc, const std::filesystem::path &path,
        vk::CommandBuffer cb, const Buffer &stagingBuffer,
        const Texture2DOptions &options = Texture2DOptions{});

    [[nodiscard]] vk::DescriptorImageInfo imageInfo() const override;
};

class Texture3D : public Texture
{
  public:
    void init(
        wheels::ScopedScratch scopeAlloc, const std::filesystem::path &path,
        ImageState initialState = ImageState::Unknown);

    [[nodiscard]] vk::DescriptorImageInfo imageInfo() const override;
};

class TextureCubemap : public Texture
{
  public:
    TextureCubemap() noexcept = default;
    ~TextureCubemap() override;

    TextureCubemap(const TextureCubemap &other) = delete;
    TextureCubemap(TextureCubemap &&other) noexcept;
    TextureCubemap &operator=(const TextureCubemap &other) = delete;
    TextureCubemap &operator=(TextureCubemap &&other) noexcept;

    void init(
        wheels::ScopedScratch scopeAlloc, const std::filesystem::path &path);

    [[nodiscard]] vk::DescriptorImageInfo imageInfo() const override;

  private:
    void copyPixels(
        wheels::ScopedScratch scopeAlloc, const Ktx &cube,
        const vk::ImageSubresourceRange &subresourceRange) const;

    vk::Sampler m_sampler;
};

#endif // PROSPER_SCENE_TEXTURE_HPP
