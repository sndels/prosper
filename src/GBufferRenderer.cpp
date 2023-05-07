#include "GBufferRenderer.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

#include "LightClustering.hpp"
#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;
using namespace wheels;

namespace
{
enum BindingSet : uint32_t
{
    CameraBindingSet = 0,
    MaterialsBindingSet = 1,
    GeometryBuffersBindingSet = 2,
    ModelInstanceTrfnsBindingSet = 3,
    BindingSetCount = 4
};

struct PCBlock
{
    uint32_t modelInstanceID{0xFFFFFFFF};
    uint32_t meshID{0xFFFFFFFF};
    uint32_t materialID{0xFFFFFFFF};
    // Not actually used but let's have it to share vert shader with forward
    // renderer
    uint32_t drawType{0};
};

} // namespace

GBufferRenderer::GBufferRenderer(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);

    printf("Creating GBufferRenderer\n");

    if (!compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
        throw std::runtime_error("GBufferRenderer shader compilation failed");

    recreate(swapConfig, camDSLayout, worldDSLayouts);
}

GBufferRenderer::~GBufferRenderer()
{
    if (_device != nullptr)
    {
        destroySwapchainRelated();

        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);
    }
}

void GBufferRenderer::recompileShaders(
    ScopedScratch scopeAlloc, const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    if (compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
    {
        destroyGraphicsPipeline();
        createGraphicsPipelines(swapConfig, camDSLayout, worldDSLayouts);
    }
}

void GBufferRenderer::recreate(
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    destroySwapchainRelated();

    createOutputs(swapConfig);
    createAttachments();
    createGraphicsPipelines(swapConfig, camDSLayout, worldDSLayouts);
}

void GBufferRenderer::record(
    vk::CommandBuffer cb, const World &world, const Camera &cam,
    const vk::Rect2D &renderArea, const uint32_t nextImage,
    Profiler *profiler) const
{
    assert(profiler != nullptr);

    {
        const auto _s = profiler->createCpuGpuScope(cb, "GBuffer");

        const StaticArray imageBarriers{
            _resources->images.albedoRoughness.transitionBarrier(ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .accessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .layout = vk::ImageLayout::eColorAttachmentOptimal,
            }),
            _resources->images.normalMetalness.transitionBarrier(ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .accessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .layout = vk::ImageLayout::eColorAttachmentOptimal,
            }),
            _resources->images.sceneDepth.transitionBarrier(ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests,
                .accessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                .layout = vk::ImageLayout::eDepthAttachmentOptimal,
            }),
            _resources->buffers.lightClusters.pointers.transitionBarrier(
                ImageState{
                    .stageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                    .accessMask = vk::AccessFlagBits2::eShaderRead,
                    .layout = vk::ImageLayout::eGeneral,
                }),
        };

        const StaticArray bufferBarriers{
            _resources->buffers.lightClusters.indicesCount.transitionBarrier(
                BufferState{
                    .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .accessMask = vk::AccessFlagBits2::eShaderRead,
                }),
            _resources->buffers.lightClusters.indices.transitionBarrier(
                BufferState{
                    .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .accessMask = vk::AccessFlagBits2::eShaderRead,
                }),
        };

        cb.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount =
                asserted_cast<uint32_t>(bufferBarriers.size()),
            .pBufferMemoryBarriers = bufferBarriers.data(),
            .imageMemoryBarrierCount =
                asserted_cast<uint32_t>(imageBarriers.size()),
            .pImageMemoryBarriers = imageBarriers.data(),
        });

        cb.beginRendering(vk::RenderingInfo{
            .renderArea = renderArea,
            .layerCount = 1,
            .colorAttachmentCount =
                asserted_cast<uint32_t>(_colorAttachments.capacity()),
            .pColorAttachments = _colorAttachments.data(),
            .pDepthAttachment = &_depthAttachment,
        });

        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

        const auto &scene = world._scenes[world._currentScene];

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[CameraBindingSet] = cam.descriptorSet(nextImage);
        descriptorSets[MaterialsBindingSet] = world._materialTexturesDS;
        descriptorSets[GeometryBuffersBindingSet] = world._geometryDS;
        descriptorSets[ModelInstanceTrfnsBindingSet] =
            scene.modelInstancesDescriptorSets[nextImage];

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, _pipelineLayout,
            0, // firstSet
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(), 0, nullptr);

        for (const auto &instance : scene.modelInstances)
        {
            const auto &model = world._models[instance.modelID];
            for (const auto &subModel : model.subModels)
            {
                const auto &material = world._materials[subModel.materialID];
                const auto &info = world._meshInfos[subModel.meshID];

                if (material.alphaMode != Material::AlphaMode::Blend)
                {
                    // TODO: Push buffers and offsets
                    const PCBlock pcBlock{
                        .modelInstanceID = instance.id,
                        .meshID = subModel.meshID,
                        .materialID = subModel.materialID,
                        .drawType = 0,
                    };
                    cb.pushConstants(
                        _pipelineLayout,
                        vk::ShaderStageFlagBits::eVertex |
                            vk::ShaderStageFlagBits::eFragment,
                        0, // offset
                        sizeof(PCBlock), &pcBlock);

                    cb.draw(info.indexCount, 1, 0, 0);
                }
            }
        }

        cb.endRendering();
    }
}

bool GBufferRenderer::compileShaders(
    ScopedScratch scopeAlloc, const World::DSLayouts &worldDSLayouts)
{
    printf("Compiling GBufferRenderer shaders\n");

    String vertDefines{scopeAlloc, 128};
    appendDefineStr(vertDefines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(vertDefines, "GEOMETRY_SET", GeometryBuffersBindingSet);
    appendDefineStr(
        vertDefines, "MODEL_INSTANCE_TRFNS_SET", ModelInstanceTrfnsBindingSet);
    const auto vertSM = _device->compileShaderModule(
        scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                      .relPath = "shader/scene.vert",
                                      .debugName = "gbufferVS",
                                      .defines = vertDefines,
                                  });

    String fragDefines{scopeAlloc, 128};
    appendDefineStr(fragDefines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(fragDefines, "MATERIALS_SET", MaterialsBindingSet);
    appendDefineStr(
        fragDefines, "NUM_MATERIAL_SAMPLERS",
        worldDSLayouts.materialSamplerCount);

    const auto fragSM = _device->compileShaderModule(
        scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                      .relPath = "shader/gbuffer.frag",
                                      .debugName = "gbuffferPS",
                                      .defines = fragDefines,
                                  });

    if (vertSM.has_value() && fragSM.has_value())
    {
        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);

        _shaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *vertSM,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *fragSM,
                .pName = "main",
            }};

        return true;
    }

    if (vertSM.has_value())
        _device->logical().destroy(*vertSM);
    if (fragSM.has_value())
        _device->logical().destroy(*fragSM);

    return false;
}

void GBufferRenderer::destroySwapchainRelated()
{
    if (_device != nullptr)
    {
        destroyGraphicsPipeline();

        _device->destroy(_resources->images.albedoRoughness);
        _device->destroy(_resources->images.normalMetalness);
        // Depth owned by Renderer

        _colorAttachments.resize(_colorAttachments.capacity(), {});
    }
}

void GBufferRenderer::destroyGraphicsPipeline()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

void GBufferRenderer::createOutputs(const SwapchainConfig &swapConfig)
{
    _resources->images.albedoRoughness = _device->createImage(ImageCreateInfo{
        .format = vk::Format::eR8G8B8A8Unorm,
        .width = swapConfig.extent.width,
        .height = swapConfig.extent.height,
        .usageFlags = vk::ImageUsageFlagBits::eColorAttachment | // Render
                      vk::ImageUsageFlagBits::eStorage,          // Shading
        .debugName = "albedoRoughness",
    });
    _resources->images.normalMetalness = _device->createImage(ImageCreateInfo{
        .format = vk::Format::eR16G16B16A16Sfloat,
        .width = swapConfig.extent.width,
        .height = swapConfig.extent.height,
        .usageFlags = vk::ImageUsageFlagBits::eColorAttachment | // Render
                      vk::ImageUsageFlagBits::eStorage,          // Shading
        .debugName = "normalMetalness",
    });
    // Depth created by Renderer
}

void GBufferRenderer::createAttachments()
{
    _colorAttachments[0] = vk::RenderingAttachmentInfo{
        .imageView = _resources->images.albedoRoughness.view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
    };
    _colorAttachments[1] = vk::RenderingAttachmentInfo{
        .imageView = _resources->images.normalMetalness.view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
    };
    _depthAttachment = vk::RenderingAttachmentInfo{
        .imageView = _resources->images.sceneDepth.view,
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{std::array{1.f, 0.f, 0.f, 0.f}},
    };
}

void GBufferRenderer::createGraphicsPipelines(
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    // Empty as we'll load vertices manually from a buffer
    const vk::PipelineVertexInputStateCreateInfo vertInputInfo;

    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };

    // TODO: Dynamic viewport state?
    const vk::Viewport viewport{
        .x = 0.f,
        .y = 0.f,
        .width = static_cast<float>(swapConfig.extent.width),
        .height = static_cast<float>(swapConfig.extent.height),
        .minDepth = 0.f,
        .maxDepth = 1.f,
    };
    const vk::Rect2D scissor{
        .offset = {0, 0},
        .extent = swapConfig.extent,
    };
    const vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor};

    const vk::PipelineRasterizationStateCreateInfo rasterizerState{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .lineWidth = 1.0,
    };

    const vk::PipelineMultisampleStateCreateInfo multisampleState{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };

    const vk::PipelineDepthStencilStateCreateInfo depthStencilState{
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = vk::CompareOp::eLess,
    };

    const vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = vk::BlendFactor::eOne,
        .dstColorBlendFactor = vk::BlendFactor::eZero,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    // TODO:
    // Make wheels::StaticArray::capacity() a static constexpr, use
    // _colorAttachments.capacity() here
    const StaticArray<vk::PipelineColorBlendAttachmentState, 2>
        colorBlendAttachments{colorBlendAttachment};
    const vk::PipelineColorBlendStateCreateInfo colorBlendState{
        .attachmentCount =
            asserted_cast<uint32_t>(colorBlendAttachments.capacity()),
        .pAttachments = colorBlendAttachments.data(),
    };

    StaticArray<vk::DescriptorSetLayout, BindingSetCount> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[CameraBindingSet] = camDSLayout;
    setLayouts[MaterialsBindingSet] = worldDSLayouts.materialTextures;
    setLayouts[GeometryBuffersBindingSet] = worldDSLayouts.geometry;
    setLayouts[ModelInstanceTrfnsBindingSet] = worldDSLayouts.modelInstances;

    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eVertex |
                      vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(PCBlock),
    };
    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = asserted_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange,
        });

    const StaticArray colorAttachmentFormats{
        _resources->images.albedoRoughness.format,
        _resources->images.normalMetalness.format,
    };

    vk::StructureChain<
        vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo>
        pipelineChain{
            vk::GraphicsPipelineCreateInfo{
                .stageCount = asserted_cast<uint32_t>(_shaderStages.size()),
                .pStages = _shaderStages.data(),
                .pVertexInputState = &vertInputInfo,
                .pInputAssemblyState = &inputAssembly,
                .pViewportState = &viewportState,
                .pRasterizationState = &rasterizerState,
                .pMultisampleState = &multisampleState,
                .pDepthStencilState = &depthStencilState,
                .pColorBlendState = &colorBlendState,
                .layout = _pipelineLayout,
            },
            vk::PipelineRenderingCreateInfo{
                .colorAttachmentCount =
                    asserted_cast<uint32_t>(colorAttachmentFormats.capacity()),
                .pColorAttachmentFormats = colorAttachmentFormats.data(),
                .depthAttachmentFormat = _resources->images.sceneDepth.format,
            }};

    {
        auto pipeline = _device->logical().createGraphicsPipeline(
            vk::PipelineCache{},
            pipelineChain.get<vk::GraphicsPipelineCreateInfo>());
        if (pipeline.result != vk::Result::eSuccess)
            throw std::runtime_error("Failed to create pbr pipeline");

        _pipeline = pipeline.value;

        _device->logical().setDebugUtilsObjectNameEXT(
            vk::DebugUtilsObjectNameInfoEXT{
                .objectType = vk::ObjectType::ePipeline,
                .objectHandle = reinterpret_cast<uint64_t>(
                    static_cast<VkPipeline>(_pipeline)),
                .pObjectName = "GBufferRenderer",
            });
    }
}