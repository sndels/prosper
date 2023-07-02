#include "DownSampler.hpp"

#define A_CPU 1
#include <ffx_a.h>
#include <ffx_spd.h>
#include <glm/glm.hpp>
#include <imgui.h>

#include <fstream>

#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;
using namespace wheels;

// TODO:
// - Device checks
//   - subgroup wave operations
//   - reading depth in compute
// - Input asserts
//   - Input dimensions <= 4096

namespace
{
struct PCBlock
{
    // needed to opt out earlier if mips are < 12
    uint32_t mips{DownSampler::sMaxMips};
    // number of total thread groups, so numWorkGroupsX * numWorkGroupsY * 1 it
    // is important to NOT take the number of slices (z dimension) into account
    // here as each slice has its own counter!
    uint32_t numWorkGroups{0};
    // These handle border control
    vec2 oneOverInputRes{};
    ivec2 inputResMinusOne{};
    ivec2 mip5ResMinusOne{};
};

const std::array<const char *, DownSampler::Operation::Count>
    sOperationDebugNames = {"DownSampler::MaxDepth"};

const std::array<const char *, DownSampler::Operation::Count> sShaderFormats = {
    "r32f",
};

const std::array<vk::Format, DownSampler::Operation::Count> sOutputFormats = {
    vk::Format::eR32Sfloat,
};

const std::array<const char *, DownSampler::Operation::Count>
    sDownSampleFunctionDefinitions = {
        R"(\
AF4 SpdLoadSourceImage(ASU2 p, AU1 slice)                                     \
{                                                                             \
    /* Sampler clamps to edge to get proper max downscale on edges */         \
    vec2 uv = p * spdConstants.oneOverInputRes;                               \
    float v = texture(sampler2D(imgSrc, srcSampler), uv).x;                   \
    return AF4(v, 0, 0 ,0);                                                   \
}                                                                             \
                                                                              \
AF4 SpdLoad(ASU2 p, AU1 slice)                                                \
{                                                                             \
    /* Clamp to edge value to get proper max downscale on edges */            \
    p = min(p, spdConstants.mip5ResMinusOne);                                 \
    return AF4(imageLoad(imgDst[5], p).x, 0, 0, 0);                           \
}                                                                             \
                                                                              \
void SpdStore(ASU2 p, AF4 value, AU1 mip, AU1 slice)                          \
{                                                                             \
    /* TODO: Does this need border controls? */                               \
    imageStore(imgDst[mip], p, AF4(value.x, 0, 0, 0));                        \
}                                                                             \
                                                                              \
AF4 SpdLoadIntermediate(AU1 x, AU1 y)                                         \
{                                                                             \
    return AF4(spdIntermediate[x][y].x, 0, 0, 0);                             \
}                                                                             \
                                                                              \
void SpdStoreIntermediate(AU1 x, AU1 y, AF4 value)                            \
{                                                                             \
    spdIntermediate[x][y].x = value.x;                                        \
}                                                                             \
                                                                              \
AF4 SpdReduce4(AF4 v0, AF4 v1, AF4 v2, AF4 v3)                                \
{                                                                             \
    return AF4(max(max(v0.x ,v1.x), max(v2.x, v3.x)), 0, 0, 0);               \
}
)",
};

vk::Extent2D getDownSampleExtent(
    const RenderResources &resources, ImageHandle inColor)
{
    const vk::Extent3D extent = resources.images.resource(inColor).extent;
    assert(extent.depth == 1);

    return vk::Extent2D{
        .width = extent.width,
        .height = extent.height,
    };
}

} // namespace

DownSampler::DownSampler(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);
    assert(staticDescriptorsAlloc != nullptr);

    printf("Creating DownSampler\n");

    if (!compileShaders(scopeAlloc.child_scope()))
        throw std::runtime_error("DownSampler shader compilation failed");

    createDescriptorSets(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    createPipelines();
    createSamplers();
}

DownSampler::~DownSampler()
{
    if (_device != nullptr)
    {
        for (const vk::Sampler s : _samplers)
            _device->logical().destroy(s);

        destroyPipelines();

        _device->logical().destroy(_descriptorSetLayout);

        for (const vk::ShaderModule sm : _compSMs)
            _device->logical().destroy(sm);
    }
}

void DownSampler::recompileShaders(ScopedScratch scopeAlloc)
{
    if (compileShaders(scopeAlloc.child_scope()))
    {
        destroyPipelines();
        createPipelines();
    }
}

bool DownSampler::compileShaders(ScopedScratch scopeAlloc)
{
    printf("Compiling DownSampler shaders\n");

    for (uint32_t op = Operation::First; op < Operation::Count; ++op)
    {
        String defines{
            scopeAlloc, 256 + strlen(sDownSampleFunctionDefinitions[op])};
        appendDefineStr(defines, "DOWN_SAMPLE_FORMAT", sShaderFormats[op]);
        defines.extend("#define DOWN_SAMPLE_FUNCTION_DEFINITIONS \\\n");
        defines.extend(sDownSampleFunctionDefinitions[op]);

        Optional<Device::ShaderCompileResult> compResult =
            _device->compileShaderModule(
                scopeAlloc.child_scope(),
                Device::CompileShaderModuleArgs{
                    .relPath = "shader/down_sampler.comp",
                    .debugName = "DownSamplerCS",
                    .defines = defines,
                });

        if (!compResult.has_value())
            return false;

        if (_compSMs.size() > op)
            _device->logical().destroy(_compSMs[op]);

        ShaderReflection &reflection = compResult->reflection;
        assert(sizeof(PCBlock) == reflection.pushConstantsBytesize());

        assert(_compSMs.size() == _shaderReflections.size());
        if (_compSMs.size() > op)
        {
            _compSMs[op] = compResult->module;
            _shaderReflections[op] = WHEELS_MOV(reflection);
        }
        else
        {
            _compSMs.push_back(compResult->module);
            _shaderReflections.push_back(WHEELS_MOV(reflection));
        }
    }

    return true;
}

DownSampler::Output DownSampler::record(
    vk::CommandBuffer cb, ImageHandle input, Operation operation,
    const uint32_t nextFrame, Profiler *profiler)
{
    assert(profiler != nullptr);
    assert(operation < Operation::Count);

    Output ret;
    {
        const vk::Extent2D extent = getDownSampleExtent(*_resources, input);

        varAU2(dispatchThreadGroupCountXY);
        varAU2(workGroupOffset);
        varAU2(numWorkGroupsAndMips);
        varAU4(rectInfo) = initAU4(0, 0, extent.width, extent.height);
        SpdSetup(
            &dispatchThreadGroupCountXY[0], &workGroupOffset[0],
            &numWorkGroupsAndMips[0], &rectInfo[0]);

        BufferHandle globalCounter;
        createResources(extent, operation, ret, globalCounter);

        const BoundResources boundResources{
            .input = input,
            .output = ret.downSampled,
            .globalCounter = globalCounter,
        };

        const uint32_t dsIndex =
            nextFrame * DownSampler::Operation::Count + operation;
        const vk::DescriptorSet descriptorSet = _descriptorSets[dsIndex];

        updateDescriptorSet(descriptorSet, boundResources, operation);

        _resources->buffers.transition(
            cb, globalCounter,
            BufferState{
                .stageMask = vk::PipelineStageFlagBits2::eCopy,
                .accessMask = vk::AccessFlagBits2::eTransferWrite,
            });

        // Start scope after the first barriers of the pass to not bleed
        // previous resource use in<to it
        auto _s =
            profiler->createCpuGpuScope(cb, sOperationDebugNames[operation]);

        // Need to reset the global counter to 0
        // ffx_spd clears it back at the end of its work but let's be safe since
        // we might be reusing a buffer from somewhere else.
        const Buffer &buffer = _resources->buffers.resource(globalCounter);
        cb.fillBuffer(buffer.handle, 0, buffer.byteSize, 0);

        recordBarriers(cb, boundResources);

        cb.bindPipeline(vk::PipelineBindPoint::eCompute, _pipelines[operation]);

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, 1,
            &descriptorSet, 0, nullptr);

        const uint32_t mips = numWorkGroupsAndMips[1];

        const ivec2 mip5ResMinusOne =
            mips < 6 ? ivec2{}
                     : ivec2{// mips[5] in shader is mip 6 relative to the input
                             asserted_cast<int32_t>((extent.width >> 6) - 1),
                             asserted_cast<int32_t>((extent.height >> 6) - 1)};

        const PCBlock pcBlock{
            .mips = mips,
            .numWorkGroups = numWorkGroupsAndMips[0],
            .oneOverInputRes = vec2{1.f} /
                               vec2{
                                   static_cast<float>(extent.width),
                                   static_cast<float>(extent.height)},
            .inputResMinusOne =
                ivec2{
                    asserted_cast<int32_t>(extent.width - 1),
                    asserted_cast<int32_t>(extent.height - 1),
                },
            .mip5ResMinusOne = mip5ResMinusOne,
        };
        cb.pushConstants(
            _pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
            sizeof(PCBlock), &pcBlock);

        cb.dispatch(
            dispatchThreadGroupCountXY[0], dispatchThreadGroupCountXY[1], 1);

        _resources->buffers.release(globalCounter);
    }

    return ret;
}

void DownSampler::destroyPipelines()
{
    for (const vk::Pipeline p : _pipelines)
        _device->logical().destroy(p);
    _pipelines.clear();
    _device->logical().destroy(_pipelineLayout);
}

void DownSampler::createDescriptorSets(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    assert(_shaderReflections.size() == Operation::Count);

    const ShaderReflection &reflection = _shaderReflections[Operation::First];
    const Array<vk::DescriptorSetLayoutBinding> layoutBindings =
        reflection.generateLayoutBindings(
            scopeAlloc, 0, vk::ShaderStageFlagBits::eCompute);

    _descriptorSetLayout = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
            .pBindings = layoutBindings.data(),
        });

#ifndef NDEBUG
    // Make sure we really can use the same layout for all sets
    // This is maybe overkill
    for (uint32_t op = Operation::First; op < Operation::Count; ++op)
    {
        const Array<vk::DescriptorSetLayoutBinding> lb =
            _shaderReflections[op].generateLayoutBindings(
                scopeAlloc, 0, vk::ShaderStageFlagBits::eCompute);
        assert(lb.size() == layoutBindings.size());
        for (size_t i = 0; i < lb.size(); ++i)
            assert(lb[i] == layoutBindings[i]);
    }
#endif // NDEBUG

    const StaticArray<vk::DescriptorSetLayout, DescriptorSets::capacity()>
        layouts{_descriptorSetLayout};
    staticDescriptorsAlloc->allocate(layouts, _descriptorSets);
}

void DownSampler::updateDescriptorSet(
    vk::DescriptorSet descriptorSet, const BoundResources &resources,
    Operation operation)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?
    const vk::DescriptorImageInfo inputInfo{
        .imageView = _resources->images.resource(resources.input).view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };

    const Span<const vk::ImageView> outputViews =
        _resources->images.subresourceViews(resources.output);
    // TODO:
    // Get subresource views into output from resource collection. It should
    // track already created subviews for images and handle destruction on exit
    StaticArray<vk::DescriptorImageInfo, sMaxMips> outputInfos;
    for (const vk::ImageView view : outputViews)
        outputInfos.push_back(vk::DescriptorImageInfo{
            .imageView = view,
            .imageLayout = vk::ImageLayout::eGeneral,
        });
    // Fill the remaining descriptors with copies of the first one so we won't
    // have unbound descriptors. We could use VK_EXT_robustness2 and null
    // descriptors, but this seems like less of a hassle since we shouldn't be
    // accessing them anyway.
    for (size_t i = outputInfos.size(); i < outputInfos.capacity(); ++i)
        outputInfos.push_back(outputInfos[0]);

    const vk::DescriptorBufferInfo counterInfo{
        .buffer = _resources->buffers.nativeHandle(resources.globalCounter),
        .range = VK_WHOLE_SIZE,
    };
    const vk::DescriptorImageInfo samplerInfo{
        .sampler = _samplers[operation],
    };

    assert(_shaderReflections.size() >= operation);
    const ShaderReflection &reflection = _shaderReflections[operation];

    const StaticArray descriptorWrites = reflection.generateDescriptorWrites<4>(
        0, descriptorSet,
        {
            Pair{0u, DescriptorInfoPtr{&inputInfo}},
            Pair{1u, DescriptorInfoPtr{outputInfos}},
            Pair{2u, DescriptorInfoPtr{&counterInfo}},
            Pair{3u, DescriptorInfoPtr{&samplerInfo}},
        });

    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void DownSampler::createPipelines()
{
    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(PCBlock),
    };

    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &_descriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange,
        });

    for (uint32_t op = Operation::First; op < Operation::Count; ++op)
    {
        const vk::ComputePipelineCreateInfo createInfo{
            .stage =
                {
                    .stage = vk::ShaderStageFlagBits::eCompute,
                    .module = _compSMs[op],
                    .pName = "main",
                },
            .layout = _pipelineLayout,
        };

        _pipelines.push_back(createComputePipeline(
            _device->logical(), createInfo, sOperationDebugNames[op]));
    }
}

void DownSampler::createResources(
    const vk::Extent2D &size, Operation operation, Output &output,
    BufferHandle &globalCounter)
{
    // Output is mips for the input
    const uint32_t width = size.width >> 1;
    const uint32_t height = size.height >> 1;
    const uint32_t mips =
        static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) +
        1;
    assert(
        mips <= DownSampler::sMaxMips &&
        "FFX SPD doesn't support generating this many mips");

    output.downSampled = _resources->images.create(
        ImageDescription{
            .format = sOutputFormats[operation],
            .width = width,
            .height = height,
            .mipCount = mips,
            .usageFlags = vk::ImageUsageFlagBits::eStorage |
                          vk::ImageUsageFlagBits::eSampled,
        },
        sOperationDebugNames[operation]);

    globalCounter = _resources->buffers.create(
        BufferDescription{
            .byteSize = sizeof(uint32_t),
            .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eTransferDst,
            .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
        },
        sOperationDebugNames[operation]);
}

void DownSampler::recordBarriers(
    vk::CommandBuffer cb, const BoundResources &resources) const
{
    const StaticArray imageBarriers{
        _resources->images.transitionBarrier(
            resources.input,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderRead,
                .layout = vk::ImageLayout::eGeneral,
            }),
        _resources->images.transitionBarrier(
            resources.output,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderRead |
                              vk::AccessFlagBits2::eShaderWrite,
                .layout = vk::ImageLayout::eGeneral,
            }),
    };

    const vk::BufferMemoryBarrier2 bufferBarrier =
        _resources->buffers.transitionBarrier(
            resources.globalCounter,
            BufferState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderStorageRead |
                              vk::AccessFlagBits2::eShaderStorageWrite,
            });

    cb.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &bufferBarrier,
        .imageMemoryBarrierCount =
            asserted_cast<uint32_t>(imageBarriers.size()),
        .pImageMemoryBarriers = imageBarriers.data(),
    });
}

void DownSampler::createSamplers()
{
    const vk::SamplerCreateInfo info{
        .magFilter = vk::Filter::eNearest,
        .minFilter = vk::Filter::eNearest,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1,
        .minLod = 0,
        .maxLod = VK_LOD_CLAMP_NONE,
    };
    _samplers.push_back(_device->logical().createSampler(info));
}
