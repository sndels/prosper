#include "RtDiSpatialReuse.hpp"

#include "../../scene/Camera.hpp"
#include "../../scene/Light.hpp"
#include "../../scene/World.hpp"
#include "../../scene/WorldRenderStructs.hpp"
#include "../../utils/Profiler.hpp"
#include "../../utils/Utils.hpp"
#include "../GBufferRenderer.hpp"
#include "../RenderResources.hpp"

using namespace glm;
using namespace wheels;

namespace
{

constexpr uint32_t sFramePeriod = 4096;

enum BindingSet : uint32_t
{
    LightsBindingSet,
    CameraBindingSet,
    StorageBindingSet,
    BindingSetCount,
};

struct PCBlock
{
    uint32_t frameIndex{0};
};

vk::Extent2D getRenderExtent(
    const RenderResources &resources, const GBufferRendererOutput &gbuffer)
{
    const vk::Extent3D targetExtent =
        resources.images.resource(gbuffer.albedoRoughness).extent;
    WHEELS_ASSERT(targetExtent.depth == 1);

    return vk::Extent2D{
        .width = targetExtent.width,
        .height = targetExtent.height,
    };
}

ComputePass::Shader shaderDefinitionCallback(
    Allocator &alloc, const WorldDSLayouts &worldDSLayouts)
{
    const size_t len = 768;
    String defines{alloc, len};
    appendDefineStr(defines, "LIGHTS_SET", LightsBindingSet);
    appendDefineStr(defines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(defines, "STORAGE_SET", StorageBindingSet);
    appendDefineStr(
        defines, "NUM_MATERIAL_SAMPLERS", worldDSLayouts.materialSamplerCount);
    PointLights::appendShaderDefines(defines);
    SpotLights::appendShaderDefines(defines);
    WHEELS_ASSERT(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/restir_di/spatial_reuse.comp",
        .debugName = String{alloc, "RtDiSpatialReuseCS"},
        .defines = WHEELS_MOV(defines),
    };
}

StaticArray<vk::DescriptorSetLayout, BindingSetCount - 1> externalDsLayouts(
    const RtDiSpatialReuse::InputDSLayouts &dsLayouts)
{
    StaticArray<vk::DescriptorSetLayout, BindingSetCount - 1> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[LightsBindingSet] = dsLayouts.world.lights;
    setLayouts[CameraBindingSet] = dsLayouts.camera;
    return setLayouts;
}

} // namespace

void RtDiSpatialReuse::init(
    ScopedScratch scopeAlloc, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    const InputDSLayouts &dsLayouts)
{
    WHEELS_ASSERT(!_initialized);
    WHEELS_ASSERT(resources != nullptr);

    _resources = resources;
    _computePass.init(
        WHEELS_MOV(scopeAlloc), staticDescriptorsAlloc,
        [&dsLayouts](Allocator &alloc)
        { return shaderDefinitionCallback(alloc, dsLayouts.world); },
        ComputePassOptions{
            .storageSetIndex = StorageBindingSet,
            .externalDsLayouts = externalDsLayouts(dsLayouts),
        });

    _initialized = true;
}

bool RtDiSpatialReuse::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    const InputDSLayouts &dsLayouts)
{
    WHEELS_ASSERT(_initialized);

    return _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles,
        [&dsLayouts](Allocator &alloc)
        { return shaderDefinitionCallback(alloc, dsLayouts.world); },
        externalDsLayouts(dsLayouts));
}

RtDiSpatialReuse::Output RtDiSpatialReuse::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const World &world,
    const Camera &cam, const Input &input, const uint32_t nextFrame,
    Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);
    WHEELS_ASSERT(profiler != nullptr);

    _frameIndex = ++_frameIndex % sFramePeriod;

    Output ret;
    {
        const vk::Extent2D renderExtent =
            getRenderExtent(*_resources, input.gbuffer);

        ret.reservoirs = _resources->images.create(
            ImageDescription{
                .format = vk::Format::eR32G32Sfloat,
                .width = renderExtent.width,
                .height = renderExtent.height,
                .usageFlags = vk::ImageUsageFlagBits::eStorage |
                              vk::ImageUsageFlagBits::eSampled,
            },
            "RtDiSpatialReuseReservoirs");

        _computePass.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = _resources->images
                                     .resource(input.gbuffer.albedoRoughness)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = _resources->images
                                     .resource(input.gbuffer.normalMetalness)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(input.gbuffer.depth).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(input.reservoirs).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(ret.reservoirs).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler = _resources->nearestSampler,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), *_resources, cb,
            Transitions{
                .images = StaticArray<ImageTransition, 5>{{
                    {input.gbuffer.albedoRoughness,
                     ImageState::ComputeShaderRead},
                    {input.gbuffer.normalMetalness,
                     ImageState::ComputeShaderRead},
                    {input.gbuffer.depth, ImageState::ComputeShaderRead},
                    {input.reservoirs, ImageState::ComputeShaderRead},
                    {ret.reservoirs, ImageState::ComputeShaderWrite},
                }},
            });

        const auto _s = profiler->createCpuGpuScope(cb, "  SpatialReuse");

        const WorldDescriptorSets &worldDSes = world.descriptorSets();
        const WorldByteOffsets &worldByteOffsets = world.byteOffsets();

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[LightsBindingSet] = worldDSes.lights;
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[StorageBindingSet] = _computePass.storageSet(nextFrame);

        const StaticArray dynamicOffsets{{
            worldByteOffsets.directionalLight,
            worldByteOffsets.pointLights,
            worldByteOffsets.spotLights,
            cam.bufferOffset(),
        }};

        const uvec3 extent =
            glm::uvec3{renderExtent.width, renderExtent.height, 1u};

        const PCBlock pcBlock{
            .frameIndex = _frameIndex,
        };

        _computePass.record(
            cb, pcBlock, extent, descriptorSets, dynamicOffsets);
    }

    return ret;
}
