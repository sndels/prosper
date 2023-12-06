#include "RtDiInitialReservoirs.hpp"

#include "../../scene/Camera.hpp"
#include "../../scene/Light.hpp"
#include "../../scene/World.hpp"
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
    LightsBindingSet = 0,
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
        .relPath = "shader/restir_di/initial_reservoirs.comp",
        .debugName = String{alloc, "RtDiInitialReservoirsCS"},
        .defines = WHEELS_MOV(defines),
    };
}

StaticArray<vk::DescriptorSetLayout, BindingSetCount - 1> externalDsLayouts(
    const RtDiInitialReservoirs::InputDSLayouts &dsLayouts)
{
    StaticArray<vk::DescriptorSetLayout, BindingSetCount - 1> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[LightsBindingSet] = dsLayouts.world.lights;
    setLayouts[CameraBindingSet] = dsLayouts.camera;
    return setLayouts;
}

} // namespace

RtDiInitialReservoirs::RtDiInitialReservoirs(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    const InputDSLayouts &dsLayouts)
: _resources{resources}
, _computePass{
      WHEELS_MOV(scopeAlloc),
      device,
      staticDescriptorsAlloc,
      [&dsLayouts](Allocator &alloc)
      { return shaderDefinitionCallback(alloc, dsLayouts.world); },
      StorageBindingSet,
      externalDsLayouts(dsLayouts)}
{
    WHEELS_ASSERT(_resources != nullptr);
}

bool RtDiInitialReservoirs::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    const InputDSLayouts &dsLayouts)
{
    return _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles,
        [&dsLayouts](Allocator &alloc)
        { return shaderDefinitionCallback(alloc, dsLayouts.world); },
        externalDsLayouts(dsLayouts));
}

RtDiInitialReservoirs::Output RtDiInitialReservoirs::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const World &world,
    const Camera &cam, const GBufferRendererOutput &gbuffer,
    const uint32_t nextFrame, Profiler *profiler)
{
    WHEELS_ASSERT(profiler != nullptr);

    _frameIndex = ++_frameIndex % sFramePeriod;

    Output ret;
    {
        const vk::Extent2D renderExtent = getRenderExtent(*_resources, gbuffer);

        ret.reservoirs = _resources->images.create(
            ImageDescription{
                .format = vk::Format::eR32G32Sfloat,
                .width = renderExtent.width,
                .height = renderExtent.height,
                .usageFlags = vk::ImageUsageFlagBits::eStorage |
                              vk::ImageUsageFlagBits::eSampled,
            },
            "RtDiInitialReservoirs");

        _computePass.updateDescriptorSet(
            WHEELS_MOV(scopeAlloc), nextFrame,
            StaticArray{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(gbuffer.albedoRoughness)
                            .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(gbuffer.normalMetalness)
                            .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(gbuffer.depth).view,
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
            });

        transition<4>(
            *_resources, cb,
            {
                {gbuffer.albedoRoughness, ImageState::ComputeShaderRead},
                {gbuffer.normalMetalness, ImageState::ComputeShaderRead},
                {gbuffer.depth, ImageState::ComputeShaderRead},
                {ret.reservoirs, ImageState::ComputeShaderWrite},
            });

        const auto _s = profiler->createCpuGpuScope(cb, "  InitialReservoirs");

        const WorldDescriptorSets &worldDSes = world.descriptorSets();
        const WorldByteOffsets &worldByteOffsets = world.byteOffsets();

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[LightsBindingSet] = worldDSes.lights;
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[StorageBindingSet] = _computePass.storageSet(nextFrame);

        const StaticArray dynamicOffsets = {
            worldByteOffsets.directionalLight,
            worldByteOffsets.pointLights,
            worldByteOffsets.spotLights,
            cam.bufferOffset(),
        };

        const uvec3 groups = glm::uvec3{
            (glm::uvec2{renderExtent.width, renderExtent.height} - 1u) / 16u +
                1u,
            1u};

        const PCBlock pcBlock{
            .frameIndex = _frameIndex,
        };

        _computePass.record(
            cb, pcBlock, groups, descriptorSets, dynamicOffsets);
    }

    return ret;
}
