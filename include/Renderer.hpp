#ifndef PROSPER_RENDERER_HPP
#define PROSPER_RENDERER_HPP

#include <functional>

#include "Camera.hpp"
#include "Device.hpp"
#include "Swapchain.hpp"
#include "World.hpp"

class Renderer {
  public:
    struct Output {
        const Image &image;
        vk::CommandBuffer commandBuffer;
    };

    struct Pipelines {
        vk::Pipeline pbr;
        vk::Pipeline pbrAlphaBlend;
        vk::Pipeline skybox;
    };

    struct PipelineLayouts {
        vk::PipelineLayout pbr;
        vk::PipelineLayout skybox;
    };

    Renderer(std::shared_ptr<Device> device, const SwapchainConfig &swapConfig,
             const vk::DescriptorSetLayout camDSLayout,
             const World::DSLayouts &worldDSLayouts);
    ~Renderer();

    Renderer(const Renderer &other) = delete;
    Renderer &operator=(const Renderer &other) = delete;

    void recreateSwapchainRelated(const SwapchainConfig &swapConfig,
                                  const vk::DescriptorSetLayout camDSLayout,
                                  const World::DSLayouts &worldDSLayouts);

    vk::Semaphore imageAvailable(const size_t frame) const;
    vk::RenderPass outputRenderpass() const;

    Output drawFrame(const World &world, const Camera &cam,
                     const vk::Rect2D &renderArea,
                     const uint32_t nextImage) const;

  private:
    void destroySwapchainRelated();
    // These also need to be recreated with Swapchain as they depend on
    // swapconfig
    void createRenderPass(const SwapchainConfig &swapConfig);
    void createFramebuffer(const SwapchainConfig &swapConfig);
    void createGraphicsPipelines(const SwapchainConfig &swapConfig,
                                 const vk::DescriptorSetLayout camDSLayout,
                                 const World::DSLayouts &worldDSLayouts);
    void createCommandBuffers(const SwapchainConfig &swapConfig);

    void updateUniformBuffers(const World &world, const Camera &cam,
                              const uint32_t nextImage) const;
    vk::CommandBuffer recordCommandBuffer(const World &world, const Camera &cam,
                                          const vk::Rect2D &renderArea,
                                          const uint32_t nextImage) const;
    void recordModelInstances(
        const vk::CommandBuffer buffer, const uint32_t nextImage,
        const std::vector<Scene::ModelInstance> &instances,
        const std::function<bool(const Mesh &)> &cullMesh) const;

    std::shared_ptr<Device> _device = nullptr;

    PipelineLayouts _pipelineLayouts;
    Pipelines _pipelines;

    vk::RenderPass _renderpass;

    std::vector<vk::CommandBuffer> _commandBuffers;

    vk::Framebuffer _fbo;
    Image _colorImage;
    Image _depthImage;
};

#endif // PROSPER_RENDERER_HPP
