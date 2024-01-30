#include "LightClustering.hpp"

#include <wheels/containers/span.hpp>
#include <wheels/containers/static_array.hpp>

#include "../gfx/VkUtils.hpp"
#include "../scene/Camera.hpp"
#include "../scene/Light.hpp"
#include "../scene/World.hpp"
#include "../scene/WorldRenderStructs.hpp"
#include "../utils/Profiler.hpp"
#include "RenderResources.hpp"

using namespace glm;
using namespace wheels;

namespace
{

constexpr uint32_t sGroupDim = 16;
constexpr uint32_t maxPointIndicesPerTile = 128;
constexpr uint32_t maxSpotIndicesPerTile = 128;

enum BindingSet : uint32_t
{
    LightsBindingSet,
    CameraBindingSet,
    LightClustersBindingSet,
    BindingSetCount,
};

struct ClusteringPCBlock
{
    uvec2 resolution;
};

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    const size_t len = 256;
    String defines{alloc, len};
    appendDefineStr(defines, "LIGHTS_SET", LightsBindingSet);
    appendDefineStr(defines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(defines, "LIGHT_CLUSTERS_SET", LightClustersBindingSet);
    PointLights::appendShaderDefines(defines);
    SpotLights::appendShaderDefines(defines);
    LightClustering::appendShaderDefines(defines);
    WHEELS_ASSERT(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/light_clustering.comp",
        .debugName = String{alloc, "LightClusteringCS"},
        .defines = WHEELS_MOV(defines),
        .groupSize = uvec3{sGroupDim, sGroupDim, 1u},
    };
}

StaticArray<vk::DescriptorSetLayout, BindingSetCount - 1> externalDsLayouts(
    const vk::DescriptorSetLayout &camDSLayout,
    const WorldDSLayouts &worldDSLayout)
{
    StaticArray<vk::DescriptorSetLayout, BindingSetCount - 1> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[LightsBindingSet] = worldDSLayout.lights;
    setLayouts[CameraBindingSet] = camDSLayout;
    return setLayouts;
}

} // namespace

void LightClustering::init(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    const vk::DescriptorSetLayout camDSLayout,
    const WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(!_initialized);
    WHEELS_ASSERT(resources != nullptr);

    _resources = resources;
    _computePass.init(
        WHEELS_MOV(scopeAlloc), device, staticDescriptorsAlloc,
        shaderDefinitionCallback,
        ComputePassOptions{
            .storageSetIndex = LightClustersBindingSet,
            .externalDsLayouts = externalDsLayouts(camDSLayout, worldDSLayouts),
            .storageStageFlags = vk::ShaderStageFlagBits::eCompute |
                                 vk::ShaderStageFlagBits::eFragment,
        });

    _initialized = true;
}

vk::DescriptorSetLayout LightClustering::descriptorSetLayout() const
{
    WHEELS_ASSERT(_initialized);

    return _computePass.storageSetLayout();
}

void LightClustering::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    const vk::DescriptorSetLayout camDSLayout,
    const WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(_initialized);

    _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback,
        externalDsLayouts(camDSLayout, worldDSLayouts));
}

LightClusteringOutput LightClustering::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const World &world,
    const Camera &cam, const vk::Extent2D &renderExtent,
    const uint32_t nextFrame, Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);

    LightClusteringOutput ret;
    {
        ret = createOutputs(renderExtent);

        _computePass.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = _resources->images.resource(ret.pointers).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{
                    _resources->texelBuffers.resource(ret.indicesCount).view},
                DescriptorInfo{
                    _resources->texelBuffers.resource(ret.indices).view},
            }});
        ret.descriptorSet = _computePass.storageSet(nextFrame);

        transition(
            WHEELS_MOV(scopeAlloc), *_resources, cb,
            Transitions{
                .images = StaticArray<ImageTransition, 1>{{
                    {ret.pointers, ImageState::ComputeShaderWrite},
                }},
                .texelBuffers = StaticArray<TexelBufferTransition, 2>{{
                    {ret.indices, BufferState::ComputeShaderWrite},
                    {ret.indicesCount, BufferState::TransferDst},
                }},
            });

        const auto _s = profiler->createCpuGpuScope(cb, "LightClustering");

        { // Reset count
            const TexelBuffer &indicesCount =
                _resources->texelBuffers.resource(ret.indicesCount);

            cb.fillBuffer(indicesCount.handle, 0, indicesCount.size, 0);

            _resources->texelBuffers.transition(
                cb, ret.indicesCount, BufferState::ComputeShaderReadWrite);
        }

        { // Main dispatch
            const ClusteringPCBlock pcBlock{
                .resolution = uvec2(renderExtent.width, renderExtent.height),
            };

            const WorldDescriptorSets &worldDSes = world.descriptorSets();
            const WorldByteOffsets &worldByteOffsets = world.byteOffsets();

            StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
                VK_NULL_HANDLE};
            descriptorSets[LightsBindingSet] = worldDSes.lights;
            descriptorSets[CameraBindingSet] = cam.descriptorSet();
            descriptorSets[LightClustersBindingSet] = ret.descriptorSet;

            const StaticArray dynamicOffsets{{
                worldByteOffsets.directionalLight,
                worldByteOffsets.pointLights,
                worldByteOffsets.spotLights,
                cam.bufferOffset(),
            }};

            const vk::Extent3D &outputExtent =
                _resources->images.resource(ret.pointers).extent;
            // Each cluster should have a separate compute group
            const uvec3 extent =
                uvec3{
                    outputExtent.width, outputExtent.height,
                    outputExtent.depth} *
                uvec3{sGroupDim, sGroupDim, 1u};

            _computePass.record(
                cb, pcBlock, extent, descriptorSets, dynamicOffsets);
        }
    }

    return ret;
}

LightClusteringOutput LightClustering::createOutputs(
    const vk::Extent2D &renderExtent)
{
    const auto pointersWidth = ((renderExtent.width - 1u) / clusterDim) + 1u;
    const auto pointersHeight = ((renderExtent.height - 1u) / clusterDim) + 1u;
    const auto pointersDepth = zSlices + 1;

    LightClusteringOutput ret;

    ret.pointers = _resources->images.create(
        ImageDescription{
            .imageType = vk::ImageType::e3D,
            .format = vk::Format::eR32G32Uint,
            .width = pointersWidth,
            .height = pointersHeight,
            .depth = pointersDepth,
            .usageFlags = vk::ImageUsageFlagBits::eSampled | // Debug
                          vk::ImageUsageFlagBits::eStorage,
        },
        "lightClusterPointers");

    ret.indicesCount = _resources->texelBuffers.create(
        TexelBufferDescription{
            .bufferDesc =
                BufferDescription{
                    .byteSize = sizeof(uint32_t),
                    .usage = vk::BufferUsageFlagBits::eTransferDst |
                             vk::BufferUsageFlagBits::eStorageTexelBuffer,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                },
            .format = vk::Format::eR32Uint,
            .supportAtomics = true,
        },
        "LightClusteringIndicesCounter");

    const vk::DeviceSize indicesSize =
        static_cast<vk::DeviceSize>(
            maxSpotIndicesPerTile + maxPointIndicesPerTile) *
        pointersWidth * pointersHeight * pointersDepth;

    ret.indices = _resources->texelBuffers.create(
        TexelBufferDescription{
            .bufferDesc =
                BufferDescription{
                    .byteSize = indicesSize * sizeof(uint16_t),
                    .usage = vk::BufferUsageFlagBits::eStorageTexelBuffer,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                },
            .format = vk::Format::eR16Uint,
        },
        "lightClusterIndices");

    return ret;
}
