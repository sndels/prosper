#include "DebugRenderer.hpp"

#include "gfx/DescriptorAllocator.hpp"
#include "gfx/VkUtils.hpp"
#include "render/LightClustering.hpp"
#include "render/RenderResources.hpp"
#include "render/RenderTargets.hpp"
#include "render/Utils.hpp"
#include "scene/Camera.hpp"
#include "utils/Logger.hpp"
#include "utils/Profiler.hpp"
#include "utils/Utils.hpp"

#include <imgui.h>

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

struct Attachments
{
    vk::RenderingAttachmentInfo color;
    vk::RenderingAttachmentInfo depth;
};

} // namespace

DebugRenderer::~DebugRenderer()
{
    // Don't check for m_initialized as we might be cleaning up after a failed
    // init.
    gDevice.logical().destroy(m_linesDSLayout);

    for (DebugLines &ls : gRenderResources.debugLines)
        gDevice.destroy(ls.buffer);

    destroyGraphicsPipeline();

    for (auto const &stage : m_shaderStages)
        gDevice.logical().destroyShaderModule(stage.module);
}

void DebugRenderer::init(
    ScopedScratch scopeAlloc, const vk::DescriptorSetLayout camDSLayout)
{
    WHEELS_ASSERT(!m_initialized);

    LOG_INFO("Creating DebugRenderer");

    if (!compileShaders(scopeAlloc.child_scope()))
        throw std::runtime_error("DebugRenderer shader compilation failed");

    for (auto i = 0u; i < MAX_FRAMES_IN_FLIGHT; ++i)
        gRenderResources.debugLines[i] = DebugLines{
            .buffer = gDevice.createBuffer(BufferCreateInfo{
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

    createDescriptorSets(scopeAlloc.child_scope());
    createGraphicsPipeline(camDSLayout);

    m_initialized = true;
}

void DebugRenderer::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    const vk::DescriptorSetLayout camDSLayout)
{
    WHEELS_ASSERT(m_initialized);

    WHEELS_ASSERT(m_vertReflection.has_value());
    WHEELS_ASSERT(m_fragReflection.has_value());
    if (!m_vertReflection->affected(changedFiles) &&
        !m_fragReflection->affected(changedFiles))
        return;

    if (compileShaders(scopeAlloc.child_scope()))
    {
        destroyGraphicsPipeline();
        createGraphicsPipeline(camDSLayout);
    }
}

void DebugRenderer::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Camera &cam,
    const RecordInOut &inOutTargets, const uint32_t nextFrame) const
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("Debug");

    {
        const vk::Rect2D renderArea = getRect2D(inOutTargets.color);

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 2>{{
                    {inOutTargets.color, ImageState::ColorAttachmentReadWrite},
                    {inOutTargets.depth, ImageState::DepthAttachmentReadWrite},
                }},
            });

        const Attachments attachments{
            .color =
                vk::RenderingAttachmentInfo{
                    .imageView =
                        gRenderResources.images->resource(inOutTargets.color)
                            .view,
                    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                    .loadOp = vk::AttachmentLoadOp::eLoad,
                    .storeOp = vk::AttachmentStoreOp::eStore,
                },
            .depth = vk::RenderingAttachmentInfo{
                .imageView =
                    gRenderResources.images->resource(inOutTargets.depth).view,
                .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eLoad,
                .storeOp = vk::AttachmentStoreOp::eStore,
            }};

        PROFILER_GPU_SCOPE_WITH_STATS(cb, "Debug");

        cb.beginRendering(vk::RenderingInfo{
            .renderArea = renderArea,
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &attachments.color,
            .pDepthAttachment = &attachments.depth,
        });

        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[GeometryBuffersBindingSet] =
            m_linesDescriptorSets[nextFrame];

        const uint32_t cameraOffset = cam.bufferOffset();

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, m_pipelineLayout,
            0, // firstSet
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(), 1, &cameraOffset);

        setViewportScissor(cb, renderArea);

        const auto &lines = gRenderResources.debugLines[nextFrame];
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
        gDevice.compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/debug_lines.vert",
                                          .debugName = "debugLinesVS",
                                          .defines = vertDefines,
                                      });

    Optional<Device::ShaderCompileResult> fragResult =
        gDevice.compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/debug_color.frag",
                                          .debugName = "debugColorPS",
                                      });

    if (vertResult.has_value() && fragResult.has_value())
    {
        for (auto const &stage : m_shaderStages)
            gDevice.logical().destroyShaderModule(stage.module);

        m_shaderStages = {{
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
        m_vertReflection = WHEELS_MOV(vertResult->reflection);
        m_fragReflection = WHEELS_MOV(fragResult->reflection);

        return true;
    }

    if (vertResult.has_value())
        gDevice.logical().destroy(vertResult->module);
    if (fragResult.has_value())
        gDevice.logical().destroy(fragResult->module);

    return false;
}

void DebugRenderer::destroyGraphicsPipeline()
{
    gDevice.logical().destroy(m_pipeline);
    gDevice.logical().destroy(m_pipelineLayout);
}

void DebugRenderer::createDescriptorSets(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(m_vertReflection.has_value());
    m_linesDSLayout = m_vertReflection->createDescriptorSetLayout(
        scopeAlloc.child_scope(), GeometryBuffersBindingSet,
        vk::ShaderStageFlagBits::eVertex);

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        m_linesDSLayout};
    const StaticArray<const char *, MAX_FRAMES_IN_FLIGHT> debugNames{
        "DebugRenderer"};
    gStaticDescriptorsAlloc.allocate(
        layouts, debugNames, m_linesDescriptorSets.mut_span());

    for (size_t i = 0; i < m_linesDescriptorSets.size(); ++i)
    {
        const StaticArray descriptorInfos{
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = gRenderResources.debugLines[i].buffer.handle,
                .range = VK_WHOLE_SIZE,
            }},
        };

        ScopedScratch loopAlloc = scopeAlloc.child_scope();
        const Array descriptorWrites =
            m_vertReflection->generateDescriptorWrites(
                loopAlloc, GeometryBuffersBindingSet, m_linesDescriptorSets[i],
                descriptorInfos);

        gDevice.logical().updateDescriptorSets(
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
    setLayouts[GeometryBuffersBindingSet] = m_linesDSLayout;

    m_pipelineLayout =
        gDevice.logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = asserted_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.data(),
        });

    const vk::PipelineColorBlendAttachmentState blendAttachment =
        opaqueColorBlendAttachment();

    // Empty as we'll load vertices manually from a buffer
    const vk::PipelineVertexInputStateCreateInfo vertInputInfo;

    m_pipeline = ::createGraphicsPipeline(
        gDevice.logical(),
        GraphicsPipelineInfo{
            .layout = m_pipelineLayout,
            .vertInputInfo = &vertInputInfo,
            .colorBlendAttachments = Span{&blendAttachment, 1},
            .shaderStages = m_shaderStages,
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
