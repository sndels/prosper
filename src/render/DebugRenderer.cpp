#include "DebugRenderer.hpp"

#include <imgui.h>

#include "../gfx/DescriptorAllocator.hpp"
#include "../gfx/VkUtils.hpp"
#include "../scene/Camera.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/Utils.hpp"
#include "LightClustering.hpp"
#include "RenderResources.hpp"
#include "RenderTargets.hpp"

using namespace glm;
using namespace wheels;

namespace
{

enum BindingSet : uint32_t
{
    CameraBindingSet,
    GeometryBuffersBindingSet,
    BindingSetCount,
};

vk::Rect2D getRenderArea(
    const RenderResources &resources,
    const DebugRenderer::RecordInOut &inOutTargets)
{
    const vk::Extent3D targetExtent =
        resources.images.resource(inOutTargets.color).extent;
    WHEELS_ASSERT(targetExtent.depth == 1);
    WHEELS_ASSERT(
        targetExtent == resources.images.resource(inOutTargets.depth).extent);

    return vk::Rect2D{
        .offset = {0, 0},
        .extent =
            {
                targetExtent.width,
                targetExtent.height,
            },
    };
}

} // namespace

DebugRenderer::~DebugRenderer()
{
    if (_device != nullptr)
    {
        _device->logical().destroy(_linesDSLayout);

        for (auto &ls : _resources->debugLines)
            _device->destroy(ls.buffer);

        destroyGraphicsPipeline();

        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);
    }
}

void DebugRenderer::init(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    const vk::DescriptorSetLayout camDSLayout)
{
    WHEELS_ASSERT(!_initialized);
    WHEELS_ASSERT(device != nullptr);
    WHEELS_ASSERT(resources != nullptr);
    WHEELS_ASSERT(staticDescriptorsAlloc != nullptr);

    _device = device;
    _resources = resources;

    printf("Creating DebugRenderer\n");

    if (!compileShaders(scopeAlloc.child_scope()))
        throw std::runtime_error("DebugRenderer shader compilation failed");

    createBuffers();
    createDescriptorSets(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    createGraphicsPipeline(camDSLayout);

    _initialized = true;
}

void DebugRenderer::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    const vk::DescriptorSetLayout camDSLayout)
{
    WHEELS_ASSERT(_initialized);

    WHEELS_ASSERT(_vertReflection.has_value());
    WHEELS_ASSERT(_fragReflection.has_value());
    if (!_vertReflection->affected(changedFiles) &&
        !_fragReflection->affected(changedFiles))
        return;

    if (compileShaders(scopeAlloc.child_scope()))
    {
        destroyGraphicsPipeline();
        createGraphicsPipeline(camDSLayout);
    }
}

void DebugRenderer::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Camera &cam,
    const RecordInOut &inOutTargets, const uint32_t nextFrame,
    Profiler *profiler) const
{
    WHEELS_ASSERT(_initialized);
    WHEELS_ASSERT(profiler != nullptr);

    {
        const vk::Rect2D renderArea = getRenderArea(*_resources, inOutTargets);

        transition(
            WHEELS_MOV(scopeAlloc), *_resources, cb,
            Transitions{
                .images = StaticArray<ImageTransition, 2>{{
                    {inOutTargets.color, ImageState::ColorAttachmentWrite},
                    {inOutTargets.depth, ImageState::DepthAttachmentReadWrite},
                }},
            });

        const Attachments attachments = createAttachments(inOutTargets);

        const auto _s = profiler->createCpuGpuScope(cb, "Debug", true);

        cb.beginRendering(vk::RenderingInfo{
            .renderArea = renderArea,
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &attachments.color,
            .pDepthAttachment = &attachments.depth,
        });

        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[GeometryBuffersBindingSet] =
            _linesDescriptorSets[nextFrame];

        const uint32_t cameraOffset = cam.bufferOffset();

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, _pipelineLayout,
            0, // firstSet
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(), 1, &cameraOffset);

        setViewportScissor(cb, renderArea);

        const auto &lines = _resources->debugLines[nextFrame];
        // No need for lines barrier, writes are mapped

        cb.draw(lines.count * 2, 1, 0, 0);

        cb.endRendering();
    }
}

bool DebugRenderer::compileShaders(ScopedScratch scopeAlloc)
{
    const size_t len = 48;
    String vertDefines{scopeAlloc, len};
    appendDefineStr(vertDefines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(vertDefines, "GEOMETRY_SET", GeometryBuffersBindingSet);
    WHEELS_ASSERT(vertDefines.size() <= len);

    Optional<Device::ShaderCompileResult> vertResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/debug_lines.vert",
                                          .debugName = "debugLinesVS",
                                          .defines = vertDefines,
                                      });

    Optional<Device::ShaderCompileResult> fragResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/debug_color.frag",
                                          .debugName = "debugColorPS",
                                      });

    if (vertResult.has_value() && fragResult.has_value())
    {
        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);

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
        _vertReflection = WHEELS_MOV(vertResult->reflection);
        _fragReflection = WHEELS_MOV(fragResult->reflection);

        return true;
    }

    if (vertResult.has_value())
        _device->logical().destroy(vertResult->module);
    if (fragResult.has_value())
        _device->logical().destroy(fragResult->module);

    return false;
}

DebugRenderer::Attachments DebugRenderer::createAttachments(
    const RecordInOut &inOutTargets) const
{
    return Attachments{
        .color =
            vk::RenderingAttachmentInfo{
                .imageView =
                    _resources->images.resource(inOutTargets.color).view,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eLoad,
                .storeOp = vk::AttachmentStoreOp::eStore,
            },
        .depth = vk::RenderingAttachmentInfo{
            .imageView = _resources->images.resource(inOutTargets.depth).view,
            .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
        }};
}

void DebugRenderer::destroyGraphicsPipeline()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

void DebugRenderer::createBuffers()
{
    for (auto i = 0u; i < MAX_FRAMES_IN_FLIGHT; ++i)
        _resources->debugLines[i] = DebugLines{
            .buffer = _device->createBuffer(BufferCreateInfo{
                .desc =
                    BufferDescription{
                        .byteSize =
                            DebugLines::sMaxLineCount * DebugLines::sLineBytes,
                        .usage = vk::BufferUsageFlagBits::eStorageBuffer,
                        .properties =
                            vk::MemoryPropertyFlagBits::eHostCoherent |
                            vk::MemoryPropertyFlagBits::eHostVisible,
                    },
                .debugName = "DebugLines",
            }),
        };
}

void DebugRenderer::createDescriptorSets(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    WHEELS_ASSERT(_vertReflection.has_value());
    _linesDSLayout = _vertReflection->createDescriptorSetLayout(
        scopeAlloc.child_scope(), *_device, GeometryBuffersBindingSet,
        vk::ShaderStageFlagBits::eVertex);

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        _linesDSLayout};
    staticDescriptorsAlloc->allocate(layouts, _linesDescriptorSets);

    for (size_t i = 0; i < _linesDescriptorSets.size(); ++i)
    {
        const StaticArray descriptorInfos{
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = _resources->debugLines[i].buffer.handle,
                .range = VK_WHOLE_SIZE,
            }},
        };

        ScopedScratch loopAlloc = scopeAlloc.child_scope();
        const Array descriptorWrites =
            _vertReflection->generateDescriptorWrites(
                loopAlloc, GeometryBuffersBindingSet, _linesDescriptorSets[i],
                descriptorInfos);

        _device->logical().updateDescriptorSets(
            asserted_cast<uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(), 0, nullptr);
    }
}

void DebugRenderer::createGraphicsPipeline(
    const vk::DescriptorSetLayout camDSLayout)
{
    StaticArray<vk::DescriptorSetLayout, BindingSetCount> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[CameraBindingSet] = camDSLayout;
    setLayouts[GeometryBuffersBindingSet] = _linesDSLayout;

    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = asserted_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.data(),
        });

    const vk::PipelineColorBlendAttachmentState blendAttachment =
        opaqueColorBlendAttachment();

    // Empty as we'll load vertices manually from a buffer
    const vk::PipelineVertexInputStateCreateInfo vertInputInfo;

    _pipeline = ::createGraphicsPipeline(
        _device->logical(),
        GraphicsPipelineInfo{
            .layout = _pipelineLayout,
            .vertInputInfo = &vertInputInfo,
            .colorBlendAttachments = Span{&blendAttachment, 1},
            .shaderStages = _shaderStages,
            .renderingInfo =
                vk::PipelineRenderingCreateInfo{
                    .colorAttachmentCount = 1,
                    .pColorAttachmentFormats = &sIlluminationFormat,
                    .depthAttachmentFormat = sDepthFormat,
                },
            .topology = vk::PrimitiveTopology::eLineList,
            .debugName = "DebugRenderer::Lines",
        });
}
