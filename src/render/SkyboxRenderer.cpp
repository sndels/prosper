#include "SkyboxRenderer.hpp"

#include "../gfx/Device.hpp"
#include "../gfx/VkUtils.hpp"
#include "../scene/Camera.hpp"
#include "../scene/World.hpp"
#include "../scene/WorldRenderStructs.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/Utils.hpp"
#include "RenderResources.hpp"
#include "RenderTargets.hpp"
#include "Utils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

enum BindingSet : uint32_t
{
    SkyboxBindingSet,
    CameraBindingSet,
    BindingSetCount,
};

struct Attachments
{
    wheels::StaticArray<vk::RenderingAttachmentInfo, 2> color;
    vk::RenderingAttachmentInfo depth;
};

} // namespace

SkyboxRenderer::~SkyboxRenderer()
{
    // Don't check for _initialized as we might be cleaning up after a failed
    // init.
    destroyGraphicsPipelines();

    for (auto const &stage : _shaderStages)
        gDevice.logical().destroyShaderModule(stage.module);
}

void SkyboxRenderer::init(
    ScopedScratch scopeAlloc, const vk::DescriptorSetLayout camDSLayout,
    const WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(!_initialized);

    printf("Creating SkyboxRenderer\n");

    if (!compileShaders(scopeAlloc.child_scope()))
        throw std::runtime_error("SkyboxRenderer shader compilation failed");

    createGraphicsPipelines(camDSLayout, worldDSLayouts);

    _initialized = true;
}

void SkyboxRenderer::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    const vk::DescriptorSetLayout camDSLayout,
    const WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(_initialized);

    WHEELS_ASSERT(_vertReflection.has_value());
    WHEELS_ASSERT(_fragReflection.has_value());
    if (!_vertReflection->affected(changedFiles) &&
        !_fragReflection->affected(changedFiles))
        return;

    if (compileShaders(scopeAlloc.child_scope()))
    {
        destroyGraphicsPipelines();
        createGraphicsPipelines(camDSLayout, worldDSLayouts);
    }
}

void SkyboxRenderer::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const World &world,
    const Camera &cam, const RecordInOut &inOutTargets,
    Profiler *profiler) const
{
    WHEELS_ASSERT(_initialized);

    PROFILER_CPU_SCOPE(profiler, "Skybox");

    {
        const vk::Rect2D renderArea = getRect2D(inOutTargets.illumination);

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 3>{{
                    {inOutTargets.illumination,
                     ImageState::ColorAttachmentReadWrite},
                    {inOutTargets.velocity,
                     ImageState::ColorAttachmentReadWrite},
                    {inOutTargets.depth, ImageState::DepthAttachmentReadWrite},
                }},
            });

        const Attachments attachments{
            .color = {{
                vk::RenderingAttachmentInfo{
                    .imageView = gRenderResources.images
                                     ->resource(inOutTargets.illumination)
                                     .view,
                    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                    .loadOp = vk::AttachmentLoadOp::eLoad,
                    .storeOp = vk::AttachmentStoreOp::eStore,
                },
                vk::RenderingAttachmentInfo{
                    .imageView =
                        gRenderResources.images->resource(inOutTargets.velocity)
                            .view,
                    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                    .loadOp = vk::AttachmentLoadOp::eLoad,
                    .storeOp = vk::AttachmentStoreOp::eStore,
                },
            }},
            .depth =
                vk::RenderingAttachmentInfo{
                    .imageView =
                        gRenderResources.images->resource(inOutTargets.depth)
                            .view,
                    .imageLayout =
                        vk::ImageLayout::eDepthStencilAttachmentOptimal,
                    .loadOp = vk::AttachmentLoadOp::eLoad,
                    .storeOp = vk::AttachmentStoreOp::eStore,
                },
        };

        PROFILER_GPU_SCOPE_WITH_STATS(profiler, cb, "Skybox");

        cb.beginRendering(vk::RenderingInfo{
            .renderArea = renderArea,
            .layerCount = 1,
            .colorAttachmentCount =
                asserted_cast<uint32_t>(attachments.color.size()),
            .pColorAttachments = attachments.color.data(),
            .pDepthAttachment = &attachments.depth,
        });

        // Skybox doesn't need to be drawn under opaque geometry but should be
        // before transparents
        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

        const WorldDescriptorSets &worldDSes = world.descriptorSets();

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[SkyboxBindingSet] = worldDSes.skybox;
        descriptorSets[CameraBindingSet] = cam.descriptorSet();

        const uint32_t camOffset = cam.bufferOffset();

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, _pipelineLayout,
            0, // firstSet
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(), 1, &camOffset);

        setViewportScissor(cb, renderArea);

        world.drawSkybox(cb);

        cb.endRendering();
    }
}

bool SkyboxRenderer::compileShaders(ScopedScratch scopeAlloc)
{
    const size_t len = 48;
    String defines{scopeAlloc, len};
    appendDefineStr(defines, "SKYBOX_SET", SkyboxBindingSet);
    appendDefineStr(defines, "CAMERA_SET", CameraBindingSet);
    WHEELS_ASSERT(defines.size() <= len);

    Optional<Device::ShaderCompileResult> vertResult =
        gDevice.compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/skybox.vert",
                                          .debugName = "skyboxVS",
                                          .defines = defines,
                                      });
    Optional<Device::ShaderCompileResult> fragResult =
        gDevice.compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/skybox.frag",
                                          .debugName = "skyboxPS",
                                          .defines = defines,
                                      });

    if (vertResult.has_value() && fragResult.has_value())
    {
        for (auto const &stage : _shaderStages)
            gDevice.logical().destroyShaderModule(stage.module);

        _vertReflection = WHEELS_MOV(vertResult->reflection);
        _fragReflection = WHEELS_MOV(fragResult->reflection);

        _shaderStages = {{
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = vertResult->module,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = fragResult->module,
                .pName = "main",
            },
        }};

        return true;
    }

    if (vertResult.has_value())
        gDevice.logical().destroy(vertResult->module);
    if (fragResult.has_value())
        gDevice.logical().destroy(fragResult->module);

    return false;
}

void SkyboxRenderer::destroyGraphicsPipelines()
{
    gDevice.logical().destroy(_pipeline);
    gDevice.logical().destroy(_pipelineLayout);
}

void SkyboxRenderer::createGraphicsPipelines(
    const vk::DescriptorSetLayout camDSLayout,
    const WorldDSLayouts &worldDSLayouts)
{
    const vk::VertexInputBindingDescription vertexBindingDescription{
        .binding = 0,
        .stride = sizeof(vec3), // Only position
        .inputRate = vk::VertexInputRate::eVertex,
    };
    const vk::VertexInputAttributeDescription vertexAttributeDescription{
        .location = 0,
        .binding = 0,
        .format = vk::Format::eR32G32B32Sfloat,
        .offset = 0,
    };
    const vk::PipelineVertexInputStateCreateInfo vertInputInfo{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertexBindingDescription,
        .vertexAttributeDescriptionCount = 1,
        .pVertexAttributeDescriptions = &vertexAttributeDescription,
    };

    StaticArray<vk::DescriptorSetLayout, BindingSetCount> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[SkyboxBindingSet] = worldDSLayouts.skybox;
    setLayouts[CameraBindingSet] = camDSLayout;

    _pipelineLayout =
        gDevice.logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = asserted_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.data(),
        });

    const StaticArray colorAttachmentFormats{{
        sIlluminationFormat,
        sVelocityFormat,
    }};

    const StaticArray<vk::PipelineColorBlendAttachmentState, 2>
        colorBlendAttachments{opaqueColorBlendAttachment()};

    _pipeline = createGraphicsPipeline(
        gDevice.logical(),
        GraphicsPipelineInfo{
            .layout = _pipelineLayout,
            .vertInputInfo = &vertInputInfo,
            .colorBlendAttachments = colorBlendAttachments,
            .shaderStages = _shaderStages,
            .renderingInfo =
                vk::PipelineRenderingCreateInfo{
                    .colorAttachmentCount = asserted_cast<uint32_t>(
                        colorAttachmentFormats.capacity()),
                    .pColorAttachmentFormats = colorAttachmentFormats.data(),
                    .depthAttachmentFormat = sDepthFormat,
                },
            .cullMode = vk::CullModeFlagBits::eNone,
            .depthCompareOp = vk::CompareOp::eGreaterOrEqual,
            .debugName = "SkyboxRenderer",
        });
}
