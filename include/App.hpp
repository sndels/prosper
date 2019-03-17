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

    void updateBuffer(Device* device, uint32_t index)
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
    // Recreates swapchain and resources tied to it
    void recreateSwapchainAndRelated();
    // Destroys resources dependent on current swapchain
    void destroySwapchainRelated();

    // Before pipeline
    void createDescriptorSetLayouts();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();

    // These need to be recreated with Swapchain
    // Before swapchain
    void createRenderPass(const SwapchainConfig& swapConfig);
    void createGraphicsPipeline(const SwapchainConfig& swapConfig);
    // After swapchain
    void createCommandBuffers();

    void createSemaphores();

    void drawFrame();
    void updateUniformBuffers(uint32_t nextImage);
    void recordCommandBuffer(uint32_t nextImage);

    Window _window; // Needs to be valid before and after everything else
    Device _device; // Needs to be valid before and after all other vk resources
    Swapchain _swapchain;
    std::vector<std::shared_ptr<Mesh>> _meshes;
    std::vector<std::shared_ptr<Texture>> _textures;
    std::vector<MeshInstance> _scene;
    Camera _cam;

    vk::DescriptorSetLayout _vkCameraDescriptorSetLayout;
    vk::DescriptorSetLayout _vkMeshInstanceDescriptorSetLayout;
    vk::DescriptorSetLayout _vkSamplerDescriptorSetLayout;
    vk::PipelineLayout _vkGraphicsPipelineLayout;

    vk::DescriptorPool _vkDescriptorPool;
    std::vector<vk::DescriptorSet> _vkCameraDescriptorSets;
    vk::DescriptorSet _vkSamplerDescriptorSet;

    vk::RenderPass _vkRenderPass;
    vk::Pipeline _vkGraphicsPipeline;

    std::vector<vk::CommandBuffer> _vkCommandBuffers;

    std::vector<vk::Semaphore> _imageAvailableSemaphores;
    std::vector<vk::Semaphore> _renderFinishedSemaphores;

};

#endif // PROSPER_APP_HPP
