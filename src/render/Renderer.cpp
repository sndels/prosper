#include "Renderer.hpp"

#include "../Allocators.hpp"
#include "../gfx/Swapchain.hpp"
#include "../scene/Camera.hpp"
#include "../scene/World.hpp"
#include "../utils/InputHandler.hpp"
#include "../utils/Logger.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/Timer.hpp"
#include "../utils/Ui.hpp"
#include "DebugRenderer.hpp"
#include "DeferredShading.hpp"
#include "ForwardRenderer.hpp"
#include "GBufferRenderer.hpp"
#include "ImGuiRenderer.hpp"
#include "ImageBasedLighting.hpp"
#include "LightClustering.hpp"
#include "MeshletCuller.hpp"
#include "RenderResources.hpp"
#include "RenderTargets.hpp"
#include "RtReference.hpp"
#include "SkyboxRenderer.hpp"
#include "TemporalAntiAliasing.hpp"
#include "TextureDebug.hpp"
#include "TextureReadback.hpp"
#include "ToneMap.hpp"
#include "dof/DepthOfField.hpp"
#include "rtdi/RtDirectIllumination.hpp"

using namespace wheels;
using namespace glm;

namespace
{

constexpr uint32_t sDrawStatsByteSize = 2 * sizeof(uint32_t);

void blitFinalComposite(
    vk::CommandBuffer cb, ImageHandle finalComposite,
    const SwapchainImage &swapImage)
{
    // Blit to support different internal rendering resolution (and color
    // format?) the future

    const StaticArray barriers{{
        *gRenderResources.images->transitionBarrier(
            finalComposite, ImageState::TransferSrc, true),
        vk::ImageMemoryBarrier2{
            // TODO:
            // What's the tight stage for this? Synchronization validation
            // complained about a hazard after color attachment write which
            // seems like an oddly specific stage for present source access to
            // happen in.
            .srcStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe,
            .srcAccessMask = vk::AccessFlags2{},
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapImage.handle,
            .subresourceRange = swapImage.subresourceRange,
        },
    }};

    cb.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = asserted_cast<uint32_t>(barriers.size()),
        .pImageMemoryBarriers = barriers.data(),
    });

    {
        PROFILER_CPU_GPU_SCOPE(cb, "BlitFinalComposite");

        const vk::ImageSubresourceLayers layers{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1};

        const vk::Extent3D &finalCompositeExtent =
            gRenderResources.images->resource(finalComposite).extent;
        WHEELS_ASSERT(finalCompositeExtent.width == swapImage.extent.width);
        WHEELS_ASSERT(finalCompositeExtent.height == swapImage.extent.height);
        const std::array offsets{
            vk::Offset3D{0, 0, 0},
            vk::Offset3D{
                asserted_cast<int32_t>(swapImage.extent.width),
                asserted_cast<int32_t>(swapImage.extent.height),
                1,
            },
        };
        const auto blit = vk::ImageBlit{
            .srcSubresource = layers,
            .srcOffsets = offsets,
            .dstSubresource = layers,
            .dstOffsets = offsets,
        };
        cb.blitImage(
            gRenderResources.images->nativeHandle(finalComposite),
            vk::ImageLayout::eTransferSrcOptimal, swapImage.handle,
            vk::ImageLayout::eTransferDstOptimal, 1, &blit,
            vk::Filter::eLinear);
    }

    {
        const vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            // TODO:
            // What's the tight stage and correct access for this?
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::ePresentSrcKHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapImage.handle,
            .subresourceRange = swapImage.subresourceRange,
        };

        cb.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier,
        });
    }
}

} // namespace

Renderer::Renderer() noexcept
: m_lightClustering{OwningPtr<LightClustering>{gAllocators.general}}
, m_forwardRenderer{OwningPtr<ForwardRenderer>{gAllocators.general}}
, m_gbufferRenderer{OwningPtr<GBufferRenderer>{gAllocators.general}}
, m_deferredShading{OwningPtr<DeferredShading>{gAllocators.general}}
, m_rtDirectIllumination{OwningPtr<RtDirectIllumination>{gAllocators.general}}
, m_rtReference{OwningPtr<RtReference>{gAllocators.general}}
, m_skyboxRenderer{OwningPtr<SkyboxRenderer>{gAllocators.general}}
, m_debugRenderer{OwningPtr<DebugRenderer>{gAllocators.general}}
, m_toneMap{OwningPtr<ToneMap>{gAllocators.general}}
, m_imguiRenderer{OwningPtr<ImGuiRenderer>{gAllocators.general}}
, m_textureDebug{OwningPtr<TextureDebug>{gAllocators.general}}
, m_depthOfField{OwningPtr<DepthOfField>{gAllocators.general}}
, m_imageBasedLighting{OwningPtr<ImageBasedLighting>{gAllocators.general}}
, m_temporalAntiAliasing{OwningPtr<TemporalAntiAliasing>{gAllocators.general}}
, m_meshletCuller{OwningPtr<MeshletCuller>{gAllocators.general}}
, m_textureReadback{OwningPtr<TextureReadback>{gAllocators.general}}
{
}

// Define here to have the definitions of the member classes available without
// including the headers in Renderer.hpp
Renderer::~Renderer() = default;

void Renderer::init(
    wheels::ScopedScratch scopeAlloc, const SwapchainConfig &swapchainConfig,
    vk::DescriptorSetLayout camDsLayout, const WorldDSLayouts &worldDsLayouts)
{
    const Timer gpuPassesInitTimer;
    m_lightClustering->init(
        scopeAlloc.child_scope(), camDsLayout, worldDsLayouts);
    m_forwardRenderer->init(
        scopeAlloc.child_scope(),
        ForwardRenderer::InputDSLayouts{
            .camera = camDsLayout,
            .lightClusters = m_lightClustering->descriptorSetLayout(),
            .world = worldDsLayouts,
        });
    m_gbufferRenderer->init(
        scopeAlloc.child_scope(), camDsLayout, worldDsLayouts);
    m_deferredShading->init(
        scopeAlloc.child_scope(),
        DeferredShading::InputDSLayouts{
            .camera = camDsLayout,
            .lightClusters = m_lightClustering->descriptorSetLayout(),
            .world = worldDsLayouts,
        });
    m_rtDirectIllumination->init(
        scopeAlloc.child_scope(), camDsLayout, worldDsLayouts);
    m_rtReference->init(scopeAlloc.child_scope(), camDsLayout, worldDsLayouts);
    m_skyboxRenderer->init(
        scopeAlloc.child_scope(), camDsLayout, worldDsLayouts);
    m_debugRenderer->init(scopeAlloc.child_scope(), camDsLayout);
    m_toneMap->init(scopeAlloc.child_scope());
    m_imguiRenderer->init(swapchainConfig);
    m_textureDebug->init(scopeAlloc.child_scope());
    m_depthOfField->init(scopeAlloc.child_scope(), camDsLayout);
    m_imageBasedLighting->init(scopeAlloc.child_scope());
    m_temporalAntiAliasing->init(scopeAlloc.child_scope(), camDsLayout);
    m_meshletCuller->init(
        scopeAlloc.child_scope(), worldDsLayouts, camDsLayout);
    m_textureReadback->init(scopeAlloc.child_scope());
    LOG_INFO("GPU pass init took %.2fs", gpuPassesInitTimer.getSeconds());
}

void Renderer::startFrame()
{
    gRenderResources.startFrame();
    m_meshletCuller->startFrame();
    m_depthOfField->startFrame();
    m_textureReadback->startFrame();

    // TODO:
    // Is this ok here? should it happen after gpu frame starts and we have the
    // next swapchain index?
    m_imguiRenderer->startFrame();
}

void Renderer::recompileShaders(
    wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDsLayout,
    const WorldDSLayouts &worldDsLayouts,
    const HashSet<std::filesystem::path> &changedFiles)
{
    LOG_INFO("Recompiling shaders");

    const Timer t;

    m_lightClustering->recompileShaders(
        scopeAlloc.child_scope(), changedFiles, camDsLayout, worldDsLayouts);
    m_forwardRenderer->recompileShaders(
        scopeAlloc.child_scope(), changedFiles,
        ForwardRenderer::InputDSLayouts{
            .camera = camDsLayout,
            .lightClusters = m_lightClustering->descriptorSetLayout(),
            .world = worldDsLayouts,
        });
    m_gbufferRenderer->recompileShaders(
        scopeAlloc.child_scope(), changedFiles, camDsLayout, worldDsLayouts);
    m_deferredShading->recompileShaders(
        scopeAlloc.child_scope(), changedFiles,
        DeferredShading::InputDSLayouts{
            .camera = camDsLayout,
            .lightClusters = m_lightClustering->descriptorSetLayout(),
            .world = worldDsLayouts,
        });
    m_rtDirectIllumination->recompileShaders(
        scopeAlloc.child_scope(), changedFiles, camDsLayout, worldDsLayouts);
    m_rtReference->recompileShaders(
        scopeAlloc.child_scope(), changedFiles, camDsLayout, worldDsLayouts);
    m_skyboxRenderer->recompileShaders(
        scopeAlloc.child_scope(), changedFiles, camDsLayout, worldDsLayouts);
    m_debugRenderer->recompileShaders(
        scopeAlloc.child_scope(), changedFiles, camDsLayout);
    m_toneMap->recompileShaders(scopeAlloc.child_scope(), changedFiles);
    m_textureDebug->recompileShaders(scopeAlloc.child_scope(), changedFiles);
    m_depthOfField->recompileShaders(
        scopeAlloc.child_scope(), changedFiles, camDsLayout);
    m_imageBasedLighting->recompileShaders(
        scopeAlloc.child_scope(), changedFiles);
    m_temporalAntiAliasing->recompileShaders(
        scopeAlloc.child_scope(), changedFiles, camDsLayout);
    m_meshletCuller->recompileShaders(
        scopeAlloc.child_scope(), changedFiles, worldDsLayouts, camDsLayout);

    LOG_INFO("Shaders recompiled in %.2fs", t.getSeconds());
}

void Renderer::recreateSwapchainAndRelated()
{
    gRenderResources.destroyResources();
}

void Renderer::recreateViewportRelated()
{
    gRenderResources.destroyResources();

    const ImVec2 viewportSize = m_imguiRenderer->centerAreaSize();
    m_viewportExtentInUi = vk::Extent2D{
        asserted_cast<uint32_t>(viewportSize.x),
        asserted_cast<uint32_t>(viewportSize.y),
    };
}

bool Renderer::drawUi(Camera &cam)
{
    ImGui::SetNextWindowPos(ImVec2{60.f, 235.f}, ImGuiCond_FirstUseEver);
    ImGui::Begin(
        "Renderer settings ", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    if (ImGui::Checkbox("Texture Debug", &m_textureDebugActive) &&
        !m_textureDebugActive)
        gRenderResources.images->clearDebug();

    bool rtDirty = false;
    // TODO: Droplist for main renderer type
    rtDirty |= ImGui::Checkbox("Reference RT", &m_referenceRt) && m_referenceRt;
    rtDirty |= ImGui::Checkbox("Depth of field (WIP)", &m_renderDoF);
    ImGui::Checkbox("Temporal Anti-Aliasing", &m_applyTaa);

    if (!m_referenceRt)
    {
        ImGui::Checkbox("Deferred shading", &m_renderDeferred);

        if (m_renderDeferred)
            rtDirty |= ImGui::Checkbox("RT direct illumination", &m_deferredRt);
    }

    if (!m_applyTaa)
        cam.setJitter(false);
    else
    {
        if (ImGui::CollapsingHeader(
                "Temporal Anti-Aliasing", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("Jitter", &m_applyJitter);
            cam.setJitter(m_applyJitter);
            m_temporalAntiAliasing->drawUi();
        }
    }

    if (ImGui::CollapsingHeader("Tone Map", ImGuiTreeNodeFlags_DefaultOpen))
        m_toneMap->drawUi();

    if (ImGui::CollapsingHeader("Renderer", ImGuiTreeNodeFlags_DefaultOpen))
    {
        rtDirty |= enumDropdown("Draw type", m_drawType, sDrawTypeNames);
        if (m_referenceRt)
            m_rtReference->drawUi();
        else
        {
            if (m_renderDeferred)
            {
                if (m_deferredRt)
                    m_rtDirectIllumination->drawUi();
            }
        }
        rtDirty |= ImGui::Checkbox("IBL", &m_applyIbl);
    }

    ImGui::End();

    return rtDirty;
}

void Renderer::render(
    wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Camera &cam,
    World &world, const vk::Rect2D &renderArea, const SwapchainImage &swapImage,
    const uint32_t nextFrame, const Options &options)
{
    // Clear stats for new frame
    DrawStats &drawStats = m_drawStats[nextFrame];
    drawStats = DrawStats{};

    if (gRenderResources.buffers->isValidHandle(m_gpuDrawStats[nextFrame]))
        gRenderResources.buffers->release(m_gpuDrawStats[nextFrame]);

    if (m_applyIbl && !m_imageBasedLighting->isGenerated())
        m_imageBasedLighting->recordGeneration(
            scopeAlloc.child_scope(), cb, world, nextFrame);

    const LightClusteringOutput lightClusters = m_lightClustering->record(
        scopeAlloc.child_scope(), cb, world, cam, renderArea.extent, nextFrame);

    const BufferHandle gpuDrawStats = gRenderResources.buffers->create(
        BufferDescription{
            .byteSize = sDrawStatsByteSize,
            .usage = vk::BufferUsageFlagBits::eTransferDst |
                     vk::BufferUsageFlagBits::eTransferSrc |
                     vk::BufferUsageFlagBits::eStorageBuffer,
            .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
        },
        "DrawStats");

    gRenderResources.buffers->transition(
        cb, gpuDrawStats, BufferState::TransferDst);
    cb.fillBuffer(
        gRenderResources.buffers->nativeHandle(gpuDrawStats), 0,
        sDrawStatsByteSize, 0);

    ImageHandle illumination;
    if (m_referenceRt)
    {
        m_rtDirectIllumination->releasePreserved();
        m_temporalAntiAliasing->releasePreserved();

        illumination =
            m_rtReference
                ->record(
                    scopeAlloc.child_scope(), cb, world, cam, renderArea,
                    RtReference::Options{
                        .depthOfField = m_renderDoF,
                        .ibl = m_applyIbl,
                        .colorDirty = options.rtDirty,
                        .drawType = m_drawType,
                    },
                    nextFrame)
                .illumination;
    }
    else
    {
        // Need to clean up after toggling rt off to not "leak" the resources
        m_rtReference->releasePreserved();

        ImageHandle velocity;
        ImageHandle depth;
        // Opaque
        if (m_renderDeferred)
        {
            const GBufferRendererOutput gbuffer = m_gbufferRenderer->record(
                scopeAlloc.child_scope(), cb, m_meshletCuller.get(), world, cam,
                renderArea, gpuDrawStats, m_drawType, nextFrame, &drawStats);

            if (m_deferredRt)
                illumination =
                    m_rtDirectIllumination
                        ->record(
                            scopeAlloc.child_scope(), cb, world, cam, gbuffer,
                            options.rtDirty, m_drawType, nextFrame)
                        .illumination;
            else
            {
                m_rtDirectIllumination->releasePreserved();

                illumination = m_deferredShading
                                   ->record(
                                       scopeAlloc.child_scope(), cb, world, cam,
                                       DeferredShading::Input{
                                           .gbuffer = gbuffer,
                                           .lightClusters = lightClusters,
                                       },
                                       nextFrame, m_applyIbl, m_drawType)
                                   .illumination;
            }

            gRenderResources.images->release(gbuffer.albedoRoughness);
            gRenderResources.images->release(gbuffer.normalMetalness);

            velocity = gbuffer.velocity;
            depth = gbuffer.depth;
        }
        else
        {
            m_rtDirectIllumination->releasePreserved();

            const ForwardRenderer::OpaqueOutput output =
                m_forwardRenderer->recordOpaque(
                    scopeAlloc.child_scope(), cb, m_meshletCuller.get(), world,
                    cam, renderArea, lightClusters, gpuDrawStats, nextFrame,
                    m_applyIbl, m_drawType, &drawStats);
            illumination = output.illumination;
            velocity = output.velocity;
            depth = output.depth;
        }

        m_skyboxRenderer->record(
            scopeAlloc.child_scope(), cb, world, cam,
            SkyboxRenderer::RecordInOut{
                .illumination = illumination,
                .velocity = velocity,
                .depth = depth,
            });

        // Transparent
        m_forwardRenderer->recordTransparent(
            scopeAlloc.child_scope(), cb, m_meshletCuller.get(), world, cam,
            ForwardRenderer::TransparentInOut{
                .illumination = illumination,
                .depth = depth,
            },
            lightClusters, gpuDrawStats, nextFrame, m_drawType, &drawStats);

        m_debugRenderer->record(
            scopeAlloc.child_scope(), cb, cam,
            DebugRenderer::RecordInOut{
                .color = illumination,
                .depth = depth,
            },
            nextFrame);

        if (options.readbackDepthPx.has_value())

            m_textureReadback->record(
                scopeAlloc.child_scope(), cb, depth, *options.readbackDepthPx,
                nextFrame);

        if (m_applyTaa)
        {
            const TemporalAntiAliasing::Output taaOutput =
                m_temporalAntiAliasing->record(
                    scopeAlloc.child_scope(), cb, cam,
                    TemporalAntiAliasing::Input{
                        .illumination = illumination,
                        .velocity = velocity,
                        .depth = depth,
                    },
                    nextFrame);

            gRenderResources.images->release(illumination);
            illumination = taaOutput.resolvedIllumination;
        }
        else
            m_temporalAntiAliasing->releasePreserved();

        // TODO:
        // Do DoF on raw illumination and have a separate stabilizing TAA pass
        // that doesn't blend foreground/background (Karis/Abadie).
        if (m_renderDoF)
        {
            const DepthOfField::Output dofOutput = m_depthOfField->record(
                scopeAlloc.child_scope(), cb, cam,
                DepthOfField::Input{
                    .illumination = illumination,
                    .depth = depth,
                },
                nextFrame);

            gRenderResources.images->release(illumination);
            illumination = dofOutput.combinedIlluminationDoF;
        }

        gRenderResources.images->release(velocity);
        gRenderResources.images->release(depth);
    }
    gRenderResources.images->release(lightClusters.pointers);
    gRenderResources.texelBuffers->release(lightClusters.indicesCount);
    gRenderResources.texelBuffers->release(lightClusters.indices);

    const ImageHandle toneMapped =
        m_toneMap->record(scopeAlloc.child_scope(), cb, illumination, nextFrame)
            .toneMapped;

    gRenderResources.images->release(illumination);

    ImageHandle finalComposite;
    if (m_textureDebugActive)
    {
        const ImVec2 size = m_imguiRenderer->centerAreaSize();
        const ImVec2 offset = m_imguiRenderer->centerAreaOffset();
        const CursorState cursor = gInputHandler.cursor();

        // Have magnifier when mouse is on (an active) debug view
        const bool uiHovered = ImGui::IsAnyItemHovered();
        const bool activeTexture = m_textureDebug->textureSelected();
        const bool cursorWithinArea =
            all(greaterThan(cursor.position, vec2(offset.x, offset.y))) &&
            all(lessThan(
                cursor.position, vec2(offset.x + size.x, offset.y + size.y)));

        Optional<vec2> cursorCoord;
        // Don't have debug magnifier when using ui that overlaps the render
        // area
        if (!uiHovered && activeTexture && cursorWithinArea)
        {
            // Also don't have magnifier when e.g. mouse look is active. Let
            // InputHandler figure out if mouse should be visible or not.
            if (!gInputHandler.mouseGesture().has_value())
            {
                // The magnifier has its own pointer so let's not mask the view
                // with the OS one.
                gInputHandler.hideCursor();
                cursorCoord = cursor.position - vec2(offset.x, offset.y);
            }
        }
        else
            gInputHandler.showCursor();

        const ImageHandle debugOutput = m_textureDebug->record(
            scopeAlloc.child_scope(), cb, renderArea.extent, cursorCoord,
            nextFrame);

        finalComposite = blitColorToFinalComposite(
            scopeAlloc.child_scope(), cb, debugOutput, swapImage.extent,
            options.drawUi);

        gRenderResources.images->release(debugOutput);
    }
    else
        finalComposite = blitColorToFinalComposite(
            scopeAlloc.child_scope(), cb, toneMapped, swapImage.extent,
            options.drawUi);

    gRenderResources.images->release(toneMapped);

    if (options.drawUi)
    {
        world.drawDeferredLoadingUi();

        if (m_textureDebugActive)
            // Draw this after so that the first frame debug is active for a new
            // texture, we draw black instead of a potentially wrong output from
            // the shared texture that wasn't protected yet
            m_textureDebug->drawUi(nextFrame);

        const vk::Rect2D backbufferArea{
            .offset = {0, 0},
            .extent = swapImage.extent,
        };
        m_imguiRenderer->endFrame(cb, backbufferArea, finalComposite);
    }

    blitFinalComposite(cb, finalComposite, swapImage);

    gRenderResources.images->release(finalComposite);

    readbackDrawStats(cb, nextFrame, gpuDrawStats);

    gRenderResources.buffers->release(gpuDrawStats);

    // Need to preserve both the new and old readback buffers. Release happens
    // after the readback is read from when nextFrame wraps around.
    for (const BufferHandle buffer : m_gpuDrawStats)
    {
        if (gRenderResources.buffers->isValidHandle(buffer))
            gRenderResources.buffers->preserve(buffer);
    }
}

const DrawStats &Renderer::drawStats(uint32_t nextFrame)
{
    DrawStats &ret = m_drawStats[nextFrame];
    const BufferHandle gpuStatsHandle = m_gpuDrawStats[nextFrame];
    if (gRenderResources.buffers->isValidHandle(gpuStatsHandle))
    {
        const uint32_t *readbackPtr = static_cast<const uint32_t *>(
            gRenderResources.buffers->resource(gpuStatsHandle).mapped);
        WHEELS_ASSERT(readbackPtr != nullptr);

        ret.drawnMeshletCount = readbackPtr[0];
        ret.rasterizedTriangleCount = readbackPtr[1];
    }
    return ret;
}

const vk::Extent2D &Renderer::viewportExtentInUi() const
{
    return m_viewportExtentInUi;
}

bool Renderer::viewportResized() const
{
    const ImVec2 viewportSize = m_imguiRenderer->centerAreaSize();
    const bool resized =
        asserted_cast<uint32_t>(viewportSize.x) != m_viewportExtentInUi.width ||
        asserted_cast<uint32_t>(viewportSize.y) != m_viewportExtentInUi.height;

    return resized;
}

vec2 Renderer::viewportOffsetInUi() const
{
    const ImVec2 offset = m_imguiRenderer->centerAreaOffset();
    return vec2{offset.x, offset.y};
}

float Renderer::lodBias() const { return m_applyTaa ? -1.f : 0.f; }

bool Renderer::rtInUse() const { return m_referenceRt || m_deferredRt; }

Optional<glm::vec4> Renderer::tryDepthReadback()
{
    return m_textureReadback->readback();
}

bool Renderer::depthAvailable() const { return !m_referenceRt; }

ImageHandle Renderer::blitColorToFinalComposite(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, ImageHandle toneMapped,
    const vk::Extent2D &swapImageExtent, bool drawUi)
{
    const ImageHandle finalComposite = gRenderResources.images->create(
        ImageDescription{
            .format = sFinalCompositeFormat,
            .width = swapImageExtent.width,
            .height = swapImageExtent.height,
            .usageFlags =
                vk::ImageUsageFlagBits::eColorAttachment | // Render
                vk::ImageUsageFlagBits::eTransferDst |     // Blit from tone
                                                           // mapped
                vk::ImageUsageFlagBits::eTransferSrc,      // Blit to swap image
        },
        "finalComposite");

    // Blit tonemapped into cleared final composite before drawing ui on top
    transition(
        WHEELS_MOV(scopeAlloc), cb,
        Transitions{
            .images = StaticArray<ImageTransition, 2>{{
                {toneMapped, ImageState::TransferSrc},
                {finalComposite, ImageState::TransferDst},
            }},
        });

    // This scope has a barrier, but that's intentional as it should contain
    // both the clear and the blit
    PROFILER_CPU_GPU_SCOPE(cb, "blitColorToFinalComposite");

    const vk::ClearColorValue clearColor{0.f, 0.f, 0.f, 0.f};
    const vk::ImageSubresourceRange subresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    cb.clearColorImage(
        gRenderResources.images->nativeHandle(finalComposite),
        vk::ImageLayout::eTransferDstOptimal, &clearColor, 1,
        &subresourceRange);

    // Memory barrier for finalComposite, layout is already correct
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags{},
        {
            vk::MemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            },
        },
        {}, {});

    const vk::ImageSubresourceLayers layers{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1};

    const vk::Extent3D toneMappedExtent =
        gRenderResources.images->resource(toneMapped).extent;
    const std::array srcOffsets{
        vk::Offset3D{0, 0, 0},
        vk::Offset3D{
            asserted_cast<int32_t>(toneMappedExtent.width),
            asserted_cast<int32_t>(toneMappedExtent.height),
            1,
        },
    };

    ivec2 dstOffset;
    ivec2 dstSize;
    if (drawUi)
    {
        const ImVec2 offset = m_imguiRenderer->centerAreaOffset();
        const ImVec2 size = m_imguiRenderer->centerAreaSize();
        dstOffset = ivec2{static_cast<int32_t>(offset.x), offset.y};
        dstSize = ivec2{size.x, size.y};
    }
    else
    {
        dstOffset = ivec2{0, 0};
        dstSize = ivec2{
            asserted_cast<int32_t>(swapImageExtent.width),
            asserted_cast<int32_t>(swapImageExtent.height),
        };
    }

    const std::array dstOffsets{
        vk::Offset3D{
            std::min(
                dstOffset.x, asserted_cast<int32_t>(swapImageExtent.width - 1)),
            std::min(
                dstOffset.y,
                asserted_cast<int32_t>(swapImageExtent.height - 1)),
            0,
        },
        vk::Offset3D{
            std::min(
                asserted_cast<int32_t>(dstOffset.x + dstSize.x),
                asserted_cast<int32_t>(swapImageExtent.width)),
            std::min(
                asserted_cast<int32_t>(dstOffset.y + dstSize.y),
                asserted_cast<int32_t>(swapImageExtent.height)),
            1,
        },
    };
    const vk::ImageBlit blit = {
        .srcSubresource = layers,
        .srcOffsets = srcOffsets,
        .dstSubresource = layers,
        .dstOffsets = dstOffsets,
    };
    cb.blitImage(
        gRenderResources.images->nativeHandle(toneMapped),
        vk::ImageLayout::eTransferSrcOptimal,
        gRenderResources.images->nativeHandle(finalComposite),
        vk::ImageLayout::eTransferDstOptimal, 1, &blit, vk::Filter::eLinear);

    return finalComposite;
}

void Renderer::readbackDrawStats(
    vk::CommandBuffer cb, uint32_t nextFrame, BufferHandle srcBuffer)
{
    BufferHandle &dstBuffer = m_gpuDrawStats[nextFrame];
    WHEELS_ASSERT(!gRenderResources.buffers->isValidHandle(dstBuffer));
    dstBuffer = gRenderResources.buffers->create(
        BufferDescription{
            .byteSize = sDrawStatsByteSize,
            .usage = vk::BufferUsageFlagBits::eTransferDst,
            .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                          vk::MemoryPropertyFlagBits::eHostCoherent,
        },
        "DrawStatsReadback");
    WHEELS_ASSERT(
        gRenderResources.buffers->resource(srcBuffer).byteSize ==
        gRenderResources.buffers->resource(dstBuffer).byteSize);

    const StaticArray barriers{{
        *gRenderResources.buffers->transitionBarrier(
            srcBuffer, BufferState::TransferSrc, true),
        *gRenderResources.buffers->transitionBarrier(
            dstBuffer, BufferState::TransferDst, true),
    }};

    cb.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = asserted_cast<uint32_t>(barriers.size()),
        .pBufferMemoryBarriers = barriers.data(),
    });

    const vk::BufferCopy region{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = sDrawStatsByteSize,
    };
    cb.copyBuffer(
        gRenderResources.buffers->nativeHandle(srcBuffer),
        gRenderResources.buffers->nativeHandle(dstBuffer), 1, &region);
}
