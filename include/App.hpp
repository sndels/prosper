#ifndef PROSPER_APP_HPP
#define PROSPER_APP_HPP

#include <optional>
#include <vector>

#include "Camera.hpp"
#include "Device.hpp"
#include "Mesh.hpp"
#include "Swapchain.hpp"
#include "Texture.hpp"
#include "Window.hpp"
#include "World.hpp"

struct MeshInstanceUniforms {
    glm::mat4 modelToWorld;
};

struct MeshInstance {
    std::shared_ptr<Mesh> mesh;
    std::vector<Buffer> uniformBuffers;
    std::vector<vk::DescriptorSet> descriptorSets;
    glm::mat4 modelToWorld;

    std::vector<vk::DescriptorBufferInfo> bufferInfos() const
    {
        std::vector<vk::DescriptorBufferInfo> infos;
        for (auto& buffer : uniformBuffers)
            infos.emplace_back(buffer.handle, 0, sizeof(MeshInstanceUniforms));

        return infos;
    }

    void updateBuffer(const Device* device, const uint32_t index) const
    {
        MeshInstanceUniforms uniforms;
        uniforms.modelToWorld = modelToWorld;

        void* data;
        device->logical().mapMemory(uniformBuffers[index].memory, 0, sizeof(MeshInstanceUniforms), {}, &data);
        memcpy(data, &uniforms, sizeof(MeshInstanceUniforms));
        device->logical().unmapMemory(uniformBuffers[index].memory);
    }
};

class App {
public:
    App() = default;
    ~App();

    App(const App& other) = delete;
    App operator=(const App& other) = delete;

    void init();
    void run();

private:
    void recreateSwapchainAndRelated();
    void destroySwapchainRelated();

    void createUniformBuffers(const uint32_t swapImageCount);
    void createDescriptorPool(const uint32_t swapImageCount);
    // DescriptorSetLayouts need to be available before pipeline
    void createDescriptorSets(const uint32_t swapImageCount);
    void createSemaphores(const uint32_t concurrentFrameCount);

    // These also need to be recreated with Swapchain as they depend on swapconfig / swapchain
    void createRenderPass(const SwapchainConfig& swapConfig);
    void createGraphicsPipeline(const SwapchainConfig& swapConfig);
    void createCommandBuffers(const SwapchainConfig& swapConfig);

    void drawFrame();
    void updateUniformBuffers(const uint32_t nextImage);
    void recordCommandBuffer(const uint32_t nextImage);

    Window _window; // Needs to be valid before and after everything else
    Device _device; // Needs to be valid before and after all other vk resources
    Swapchain _swapchain;
    World _world;
    std::vector<std::shared_ptr<Mesh>> _meshes;
    std::vector<MeshInstance> _scene;
    Camera _cam;

    vk::DescriptorSetLayout _vkMeshInstanceDescriptorSetLayout;
    vk::PipelineLayout _vkGraphicsPipelineLayout;

    vk::DescriptorPool _vkDescriptorPool;

    vk::RenderPass _vkRenderPass;
    vk::Pipeline _vkGraphicsPipeline;

    std::vector<vk::CommandBuffer> _vkCommandBuffers;

    std::vector<vk::Semaphore> _imageAvailableSemaphores;
    std::vector<vk::Semaphore> _renderFinishedSemaphores;

};

#endif // PROSPER_APP_HPP
