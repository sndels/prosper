#include "GBufferRenderer.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

#include "../gfx/VkUtils.hpp"
#include "../utils/Utils.hpp"
#include "LightClustering.hpp"
#include "RenderTargets.hpp"

using namespace glm;
using namespace wheels;

namespace
{

const vk::Format sAlbedoRoughnessFormat = vk::Format::eR8G8B8A8Unorm;
const vk::Format sNormalMetalnessFormat = vk::Format::eR16G16B16A16Sfloat;

enum BindingSet : uint32_t
{
    CameraBindingSet = 0,
    MaterialDatasBindingSet = 1,
    MaterialTexturesBindingSet = 2,
    GeometryBuffersBindingSet = 3,
    ModelInstanceTrfnsBindingSet = 4,
    BindingSetCount = 5
};

struct PCBlock
{
    uint32_t modelInstanceID{0xFFFFFFFF};
    uint32_t meshID{0xFFFFFFFF};
    uint32_t materialID{0xFFFFFFFF};
};

} // namespace

GBufferRenderer::GBufferRenderer(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    WHEELS_ASSERT(_device != nullptr);
    WHEELS_ASSERT(_resources != nullptr);

    printf("Creating GBufferRenderer\n");

    if (!compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
        throw std::runtime_error("GBufferRenderer shader compilation failed");

    createGraphicsPipelines(camDSLayout, worldDSLayouts);
}

GBufferRenderer::~GBufferRenderer()
{
    if (_device != nullptr)
    {
        destroyGraphicsPipeline();

        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);
    }
}

void GBufferRenderer::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(_vertReflection.has_value());
    WHEELS_ASSERT(_fragReflection.has_value());
    if (!_vertReflection->affected(changedFiles) &&
        !_fragReflection->affected(changedFiles))
        return;

    if (compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
    {
        destroyGraphicsPipeline();
        createGraphicsPipelines(camDSLayout, worldDSLayouts);
    }
}

GBufferRenderer::Output GBufferRenderer::record(
    vk::CommandBuffer cb, const World &world, const Camera &cam,
    const vk::Rect2D &renderArea, const uint32_t nextFrame, Profiler *profiler)
{
    WHEELS_ASSERT(profiler != nullptr);

    Output ret;
    {
        ret = createOutputs(renderArea.extent);

        recordBarriers(cb, ret);

        const Attachments attachments = createAttachments(ret);

        const auto _s = profiler->createCpuGpuScope(cb, "GBuffer", true);

        cb.beginRendering(vk::RenderingInfo{
            .renderArea = renderArea,
            .layerCount = 1,
            .colorAttachmentCount =
                asserted_cast<uint32_t>(attachments.color.size()),
            .pColorAttachments = attachments.color.data(),
            .pDepthAttachment = &attachments.depth,
        });

        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

        const auto &scene = world._scenes[world._currentScene];

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[MaterialDatasBindingSet] =
            world._materialDatasDSs[nextFrame];
        descriptorSets[MaterialTexturesBindingSet] = world._materialTexturesDS;
        descriptorSets[GeometryBuffersBindingSet] = world._geometryDS;
        descriptorSets[ModelInstanceTrfnsBindingSet] =
            scene.modelInstancesDescriptorSet;

        const StaticArray dynamicOffsets{
            cam.bufferOffset(),
            world._modelInstanceTransformsByteOffset,
        };

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, _pipelineLayout,
            0, // firstSet
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(),
            asserted_cast<uint32_t>(dynamicOffsets.size()),
            dynamicOffsets.data());

        setViewportScissor(cb, renderArea);

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

    return ret;
}

bool GBufferRenderer::compileShaders(
    ScopedScratch scopeAlloc, const World::DSLayouts &worldDSLayouts)
{
    printf("Compiling GBufferRenderer shaders\n");

    const size_t vertDefsLen = 128;
    String vertDefines{scopeAlloc, vertDefsLen};
    appendDefineStr(vertDefines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(vertDefines, "GEOMETRY_SET", GeometryBuffersBindingSet);
    appendDefineStr(
        vertDefines, "MODEL_INSTANCE_TRFNS_SET", ModelInstanceTrfnsBindingSet);
    appendDefineStr(vertDefines, "USE_GBUFFER_PC");
    WHEELS_ASSERT(vertDefines.size() <= vertDefsLen);

    Optional<Device::ShaderCompileResult> vertResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/forward.vert",
                                          .debugName = "gbufferVS",
                                          .defines = vertDefines,
                                      });

    const size_t fragDefsLen = 128;
    String fragDefines{scopeAlloc, fragDefsLen};
    appendDefineStr(fragDefines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(fragDefines, "MATERIAL_DATAS_SET", MaterialDatasBindingSet);
    appendDefineStr(
        fragDefines, "MATERIAL_TEXTURES_SET", MaterialTexturesBindingSet);
    appendDefineStr(
        fragDefines, "NUM_MATERIAL_SAMPLERS",
        worldDSLayouts.materialSamplerCount);
    WHEELS_ASSERT(fragDefines.size() <= fragDefsLen);

    Optional<Device::ShaderCompileResult> fragResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/gbuffer.frag",
                                          .debugName = "gbuffferPS",
                                          .defines = fragDefines,
                                      });

    if (vertResult.has_value() && fragResult.has_value())
    {
        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);

        _vertReflection = WHEELS_MOV(vertResult->reflection);
        WHEELS_ASSERT(
            sizeof(PCBlock) == _vertReflection->pushConstantsBytesize());

        _fragReflection = WHEELS_MOV(fragResult->reflection);
        WHEELS_ASSERT(
            sizeof(PCBlock) == _fragReflection->pushConstantsBytesize());

        _shaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = vertResult->module,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = fragResult->module,
                .pName = "main",
            }};

        return true;
    }

    if (vertResult.has_value())
        _device->logical().destroy(vertResult->module);
    if (fragResult.has_value())
        _device->logical().destroy(fragResult->module);

    return false;
}

void GBufferRenderer::destroyGraphicsPipeline()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

GBufferRenderer::Output GBufferRenderer::createOutputs(
    const vk::Extent2D &size) const
{
    return Output{
        .albedoRoughness = _resources->images.create(
            ImageDescription{
                .format = sAlbedoRoughnessFormat,
                .width = size.width,
                .height = size.height,
                .usageFlags =
                    vk::ImageUsageFlagBits::eSampled |         // Debug
                    vk::ImageUsageFlagBits::eColorAttachment | // Render
                    vk::ImageUsageFlagBits::eStorage,          // Shading
            },
            "albedoRoughness"),
        .normalMetalness = _resources->images.create(
            ImageDescription{
                .format = sNormalMetalnessFormat,
                .width = size.width,
                .height = size.height,
                .usageFlags =
                    vk::ImageUsageFlagBits::eSampled |         // Debug
                    vk::ImageUsageFlagBits::eColorAttachment | // Render
                    vk::ImageUsageFlagBits::eStorage,          // Shading
            },
            "normalMetalness"),
        .depth = createDepth(*_device, *_resources, size, "depth"),
    };
}

GBufferRenderer::Attachments GBufferRenderer::createAttachments(
    const Output &output) const
{
    return Attachments{
        .color =
            {vk::RenderingAttachmentInfo{
                 .imageView =
                     _resources->images.resource(output.albedoRoughness).view,
                 .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                 .loadOp = vk::AttachmentLoadOp::eClear,
                 .storeOp = vk::AttachmentStoreOp::eStore,
                 .clearValue = vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
             },
             vk::RenderingAttachmentInfo{
                 .imageView =
                     _resources->images.resource(output.normalMetalness).view,
                 .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                 .loadOp = vk::AttachmentLoadOp::eClear,
                 .storeOp = vk::AttachmentStoreOp::eStore,
                 .clearValue = vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
             }},
        .depth =
            vk::RenderingAttachmentInfo{
                .imageView = _resources->images.resource(output.depth).view,
                .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
            },
    };
}

void GBufferRenderer::recordBarriers(
    vk::CommandBuffer cb, const Output &output) const
{
    transition<3>(
        *_resources, cb,
        {
            {output.albedoRoughness, ImageState::ColorAttachmentWrite},
            {output.normalMetalness, ImageState::ColorAttachmentWrite},
            {output.depth, ImageState::DepthAttachmentReadWrite},
        });
}

void GBufferRenderer::createGraphicsPipelines(
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    StaticArray<vk::DescriptorSetLayout, BindingSetCount> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[CameraBindingSet] = camDSLayout;
    setLayouts[MaterialDatasBindingSet] = worldDSLayouts.materialDatas;
    setLayouts[MaterialTexturesBindingSet] = worldDSLayouts.materialTextures;
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
        sAlbedoRoughnessFormat,
        sNormalMetalnessFormat,
    };

    const StaticArray<vk::PipelineColorBlendAttachmentState, 2>
        colorBlendAttachments{opaqueColorBlendAttachment()};

    // Empty as we'll load vertices manually from a buffer
    const vk::PipelineVertexInputStateCreateInfo vertInputInfo;

    _pipeline = createGraphicsPipeline(
        _device->logical(), vk::PrimitiveTopology::eTriangleList,
        _pipelineLayout, vertInputInfo, vk::CullModeFlagBits::eBack,
        vk::CompareOp::eGreater, colorBlendAttachments, _shaderStages,
        vk::PipelineRenderingCreateInfo{
            .colorAttachmentCount =
                asserted_cast<uint32_t>(colorAttachmentFormats.capacity()),
            .pColorAttachmentFormats = colorAttachmentFormats.data(),
            .depthAttachmentFormat = sDepthFormat,
        },
        "GBufferRenderer");
}
