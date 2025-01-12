#include "ShaderReflection.hpp"

#include "Device.hpp"
#include "utils/Utils.hpp"

#include <algorithm>
#include <cstring>
#include <spirv.hpp>
#include <stdexcept>
#include <variant>
#include <wheels/containers/array.hpp>
#include <wheels/containers/optional.hpp>

using namespace wheels;

namespace
{

struct SpvBool;
struct SpvInt;
struct SpvFloat;
struct SpvVector;
struct SpvMatrix;
struct SpvImage;
struct SpvSampledImage;
struct SpvSampler;
struct SpvArray;
struct SpvRuntimeArray;
struct SpvStruct;
struct SpvPointer;
struct SpvAccelerationStructure;
struct SpvConstantU32;
struct SpvVariable;
struct SpvSpecializationConstant;

// SpvVariable is not a really a type-type, but it is a type of result
using SpvType = std::variant<
    SpvBool, SpvInt, SpvFloat, SpvVector, SpvMatrix, SpvImage, SpvSampledImage,
    SpvSampler, SpvRuntimeArray, SpvArray, SpvStruct, SpvPointer,
    SpvAccelerationStructure, SpvConstantU32, SpvVariable,
    SpvSpecializationConstant>;

// From https://en.cppreference.com/w/cpp/utility/variant/visit
template <class... Ts> struct overloaded : Ts...
{
    using Ts::operator()...;
};
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

const uint32_t sUninitialized = 0xFFFF'FFFF;

struct SpvBool
{
};

struct SpvInt
{
    uint32_t width{sUninitialized};
    bool isSigned{true};
};

struct SpvFloat
{
    uint32_t width{sUninitialized};
};

struct SpvVector
{
    uint32_t componentId{sUninitialized};
    uint32_t componentCount{0};
};

struct SpvMatrix
{
    uint32_t columnId{sUninitialized};
    uint32_t columnCount{0};
};

struct SpvImage
{
    spv::Dim dimensionality{spv::DimMax};
    uint32_t sampled{sUninitialized};
};

struct SpvSampledImage
{
};

struct SpvSampler
{
};

struct SpvArray
{
    uint32_t elementTypeId{sUninitialized};
    uint32_t length{sUninitialized};
};

struct SpvRuntimeArray
{
    uint32_t elementTypeId{sUninitialized};
};

struct MemberDecorations
{
    uint32_t offset{sUninitialized};
    uint32_t matrixStride{sUninitialized};
};

struct SpvStruct
{
    Array<uint32_t> memberTypeIds;
    Array<MemberDecorations> memberDecorations;
};
struct SpvPointer
{
    uint32_t typeId{sUninitialized};
    spv::StorageClass storageClass{spv::StorageClassMax};
};

struct SpvAccelerationStructure
{
};

// Can also hold 8bit and 16bit values
struct SpvConstantU32
{
    uint32_t value{sUninitialized};
};

struct SpvVariable
{
    uint32_t typeId{sUninitialized};
    spv::StorageClass storageClass{spv::StorageClassMax};
};

struct SpvSpecializationConstant
{
    // ID comes from a decoration but all specialization constants have it so
    // let's store it here
    uint32_t id{sUninitialized};
    uint32_t size{sUninitialized};
};

struct Decorations
{
    uint32_t descriptorSet{sUninitialized};
    uint32_t binding{sUninitialized};
};

// Only valid until the bytecode is freed
struct SpvResult
{
    const char *name{nullptr};
    Optional<SpvType> type;
    Decorations decorations;
};

const size_t firstOpOffset = 5;

void firstPass(
    Allocator &alloc, Span<const uint32_t> words, Span<SpvResult> results,
    uint32_t &pushConstantMetadataId)
{
    // Collect names and types
    size_t opFirstWord = firstOpOffset;
    while (opFirstWord < words.size())
    {
        const uint16_t opWordCount =
            static_cast<uint16_t>(words[opFirstWord] >> 16);
        const uint16_t op = static_cast<uint16_t>(words[opFirstWord] & 0xFFFF);
        const Span<const uint32_t> args =
            opWordCount > 1 ? Span{&words[opFirstWord + 1], opWordCount - 1u}
                            : Span<const uint32_t>{};

        switch (op)
        {
        case spv::OpName:
        {
            const uint32_t result = args[0];
            const char *name = reinterpret_cast<const char *>(&args[1]);

            results[result].name = name;
        }
        break;
        case spv::OpTypeBool:
        {
            const uint32_t result = args[0];

            results[result].type.emplace(SpvBool{});
        }
        break;
        case spv::OpTypeInt:
        {
            const uint32_t result = args[0];
            const uint32_t width = args[1];
            const uint32_t signedness = args[2];
            WHEELS_ASSERT(signedness == 0 || signedness == 1);

            results[result].type.emplace(SpvInt{
                .width = width,
                .isSigned = signedness == 1,
            });
        }
        break;
        case spv::OpTypeFloat:
        {
            const uint32_t result = args[0];
            const uint32_t width = args[1];

            results[result].type.emplace(SpvFloat{
                .width = width,
            });
        }
        break;
        case spv::OpTypeVector:
        {
            const uint32_t result = args[0];
            const uint32_t componentType = args[1];
            const uint32_t componentCount = args[2];

            results[result].type.emplace(SpvVector{
                .componentId = componentType,
                .componentCount = componentCount,
            });
        }
        break;
        case spv::OpTypeMatrix:
        {
            const uint32_t result = args[0];
            const uint32_t columnType = args[1];
            const uint32_t columnCount = args[2];

            results[result].type.emplace(SpvMatrix{
                .columnId = columnType,
                .columnCount = columnCount,
            });
        }
        break;
        case spv::OpTypeImage:
        {
            const uint32_t result = args[0];
            // const uint32_t sampledTypeId= args[1];
            const spv::Dim dimensionality = static_cast<spv::Dim>(args[2]);
            // const uint32_t depth = args[3];
            // const uint32_t arrayed = args[4];
            // const uint32_t multiSampled = args[5];
            const uint32_t sampled = args[6];
            // const spv::ImageFormat format =
            //     static_cast<spv::ImageFormat>(args[7]);

            results[result].type.emplace(SpvImage{
                .dimensionality = dimensionality,
                .sampled = sampled,
            });
        }
        break;
        case spv::OpTypeSampler:
        {
            const uint32_t result = args[0];

            results[result].type.emplace(SpvSampler{});
        }
        break;
        case spv::OpTypeSampledImage:
        {
            const uint32_t result = args[0];
            // const uint32_t imageTypeId = args[1];

            results[result].type.emplace(SpvSampledImage{});
        }
        break;
        case spv::OpTypeStruct:
        {
            const uint32_t result = args[0];
            const uint32_t memberCount = opWordCount - 2;

            SpvStruct spvStruct{
                .memberTypeIds = Array<uint32_t>{alloc, memberCount},
                .memberDecorations = Array<MemberDecorations>{alloc},
            };

            for (uint32_t i = 1; i <= memberCount; ++i)
                spvStruct.memberTypeIds.push_back(args[i]);

            spvStruct.memberDecorations.resize(memberCount);

            results[result].type.emplace(WHEELS_MOV(spvStruct));
        }
        break;
        case spv::OpTypeArray:
        {
            const uint32_t result = args[0];
            const uint32_t elementType = args[1];
            const uint32_t lengthId = args[2];

            const SpvResult &lengthResult = results[lengthId];
            if (!lengthResult.type.has_value())
                // Arrays that use specialization constants won't have a size
                break;
            const SpvConstantU32 *length =
                std::get_if<SpvConstantU32>(&*lengthResult.type);
            WHEELS_ASSERT(length != nullptr);
            WHEELS_ASSERT(length->value != sUninitialized);

            results[result].type.emplace(SpvArray{
                .elementTypeId = elementType,
                .length = length->value,
            });
        }
        break;
        case spv::OpTypeRuntimeArray:
        {
            const uint32_t result = args[0];
            const uint32_t elementTypeId = args[1];

            results[result].type.emplace(SpvRuntimeArray{
                .elementTypeId = elementTypeId,
            });
        }
        break;
        case spv::OpTypePointer:
        {
            const uint32_t result = args[0];
            const spv::StorageClass storageClass =
                static_cast<spv::StorageClass>(args[1]);
            const uint32_t typeId = args[2];

            if (storageClass == spv::StorageClassPushConstant)
            {
                const Optional<SpvType> &type = results[typeId].type;
                // Accessors into PC struct members are also this storage class,
                // so let's just pick the struct
                if (type.has_value() &&
                    std::holds_alternative<SpvStruct>(*type))
                {
                    // This probably fires if we have structs within the push
                    // constants struct, if that's even possible
                    WHEELS_ASSERT(
                        pushConstantMetadataId == sUninitialized &&
                        "Unexpected second push constant struct pointer");
                    pushConstantMetadataId = typeId;
                }
            }

            results[result].type.emplace(SpvPointer{
                .typeId = typeId,
                .storageClass = storageClass,
            });
        }
        break;
        case spv::OpConstant:
        {
            const uint32_t typeId = args[0];
            const uint32_t result = args[1];

            const SpvResult &type = results[typeId];
            WHEELS_ASSERT(type.type.has_value());

            if (const SpvInt *spvInt = std::get_if<SpvInt>(&*type.type);
                spvInt != nullptr)
            {
                if (!spvInt->isSigned && spvInt->width == 32)
                    results[result].type.emplace(SpvConstantU32{
                        .value = args[2],
                    });
            }
        }
        break;
        case spv::OpSpecConstantTrue:
        case spv::OpSpecConstantFalse:
        {
            const uint32_t typeId = args[0];
            const uint32_t result = args[1];

            const SpvResult &type = results[typeId];
            WHEELS_ASSERT(type.type.has_value());
            WHEELS_ASSERT(std::get_if<SpvBool>(&*type.type) != nullptr);

            results[result].type.emplace(SpvSpecializationConstant{
                .size = sizeof(VkBool32),
            });
        }
        break;
        case spv::OpSpecConstant:
        {
            const uint32_t typeId = args[0];
            const uint32_t result = args[1];

            const SpvResult &type = results[typeId];
            WHEELS_ASSERT(type.type.has_value());
            SpvSpecializationConstant constant;

            if (const SpvBool *boolPtr = std::get_if<SpvBool>(&*type.type);
                boolPtr != nullptr)
            {
                constant.size = sizeof(VkBool32);
            }
            else if (const SpvInt *intPtr = std::get_if<SpvInt>(&*type.type);
                     intPtr != nullptr)
            {
                if (intPtr->width != 32)
                    throw std::runtime_error(
                        "Only 32bit integers are supported in specialization "
                        "constants");
                constant.size = intPtr->width / 8;
            }
            else if (const SpvFloat *floatPtr =
                         std::get_if<SpvFloat>(&*type.type);
                     floatPtr != nullptr)
            {
                if (floatPtr->width != 32)
                    throw std::runtime_error("Only 32bit floats are supported "
                                             "in specialization constants");
                constant.size = floatPtr->width / 8;
            }

            if (constant.size == sUninitialized)
                throw std::runtime_error(
                    "Unsupported specialization constants type");

            results[result].type.emplace(constant);
        }
        break;
        case spv::OpSpecConstantComposite:
        {
            throw std::runtime_error(
                "Composite specialization constants are not supported");
        }
        break;
        case spv::OpSpecConstantOp:
        {
            // Ignore these, they won't be ones that are given as specialization
            // map entries
        }
        break;
        case spv::OpVariable:
        {
            const uint32_t typeId = args[0];
            const uint32_t result = args[1];
            const spv::StorageClass storageClass =
                static_cast<spv::StorageClass>(args[2]);

            results[result].type.emplace(SpvVariable{
                .typeId = typeId,
                .storageClass = storageClass,
            });
        }
        break;
        case spv::OpTypeAccelerationStructureKHR:
        {
            const uint32_t result = args[0];

            results[result].type.emplace(SpvAccelerationStructure{});
        }
        break;
        default:
            break;
        }
        opFirstWord += opWordCount;
    }
}

void secondPass(Span<const uint32_t> words, Span<SpvResult> results)
{
    // Collect decorations
    size_t opFirstWord = firstOpOffset;
    while (opFirstWord < words.size())
    {
        const uint16_t opWordCount =
            static_cast<uint16_t>(words[opFirstWord] >> 16);
        const uint16_t op = static_cast<uint16_t>(words[opFirstWord] & 0xFFFF);
        const Span<const uint32_t> args =
            opWordCount > 1 ? Span{&words[opFirstWord + 1], opWordCount - 1u}
                            : Span<const uint32_t>{};

        switch (op)
        {
        case spv::OpDecorate:
        {
            const uint32_t resultId = args[0];
            const uint32_t decoration = static_cast<spv::Decoration>(args[1]);

            SpvResult &result = results[resultId];

            switch (decoration)
            {
            case spv::DecorationSpecId:
                std::get<SpvSpecializationConstant>(*result.type).id = args[2];
                break;
            case spv::DecorationDescriptorSet:
                result.decorations.descriptorSet = args[2];
                break;
            case spv::DecorationBinding:
                result.decorations.binding = args[2];
                break;
            default:
                break;
            }
        }
        break;
        case spv::OpMemberDecorate:
        {
            const uint32_t resultId = args[0];
            const uint32_t memberIndex = args[1];
            const uint32_t decoration = static_cast<spv::Decoration>(args[2]);

            SpvResult &result = results[resultId];
            if (result.type.has_value())
            {
                SpvStruct *spvStruct = std::get_if<SpvStruct>(&*result.type);
                WHEELS_ASSERT(spvStruct != nullptr);

                switch (decoration)
                {
                case spv::DecorationOffset:
                    spvStruct->memberDecorations[memberIndex].offset = args[3];
                    break;
                case spv::DecorationMatrixStride:
                    spvStruct->memberDecorations[memberIndex].matrixStride =
                        args[3];
                    break;
                default:
                    break;
                }
            }
        }
        break;
        default:
            break;
        }
        opFirstWord += opWordCount;
    }
}

// Takes in the member and its decorations from the parent struct
// Returns the raw size of the member, without padding to alignment
uint32_t memberBytesize(
    const Optional<SpvType> &type, const MemberDecorations &memberDecorations,
    Span<const SpvResult> results)
{
    WHEELS_ASSERT(
        type.has_value() && "Unimplemented member type, probably OpTypeArray");

    return std::visit(
        overloaded{
            [](const SpvBool &) -> uint32_t { return sizeof(VkBool32); },
            [](const SpvInt &v) -> uint32_t { return v.width / 8; },
            [](const SpvFloat &v) -> uint32_t { return v.width / 8; },
            [&results](const SpvVector &v) -> uint32_t
            {
                const SpvResult &componentResult = results[v.componentId];
                const uint32_t componentBytesize = memberBytesize(
                    componentResult.type, MemberDecorations{}, results);

                return componentBytesize * v.componentCount;
            },
            [&memberDecorations](const SpvMatrix &v) -> uint32_t
            {
                WHEELS_ASSERT(memberDecorations.matrixStride != sUninitialized);
                return memberDecorations.matrixStride * v.columnCount;
            },
            [&results](const SpvStruct &v) -> uint32_t
            {
                const uint32_t lastMemberId = v.memberTypeIds.back();
                const MemberDecorations &lastMemberDecorations =
                    v.memberDecorations.back();

                const SpvResult &lastMemberResult = results[lastMemberId];

                const uint32_t lastMemberBytesize = memberBytesize(
                    lastMemberResult.type, lastMemberDecorations, results);

                WHEELS_ASSERT(lastMemberDecorations.offset != sUninitialized);
                const uint32_t bytesize =
                    lastMemberDecorations.offset + lastMemberBytesize;

                return bytesize;
            },
            [](const auto &) -> uint32_t
            {
                WHEELS_ASSERT(!"Unimplemented");
                return 0;
            }},
        *type);
}

uint32_t getPushConstantsBytesize(
    Span<const SpvResult> results, uint32_t metadataId)
{
    const SpvResult &pcResult = results[metadataId];

    return memberBytesize(pcResult.type, MemberDecorations{}, results);
}

vk::DescriptorType imageDescriptorType(const SpvImage &image)
{
    switch (image.dimensionality)
    {
    case spv::Dim1D:
    case spv::Dim2D:
    case spv::Dim3D:
    case spv::DimCube:
        if (image.sampled == 1)
            return vk::DescriptorType::eSampledImage;
        else
        {
            WHEELS_ASSERT(
                image.sampled == 2 &&
                "Sampled yes/no has to be known at shader "
                "compile time");
            return vk::DescriptorType::eStorageImage;
        }
        break;
    case spv::DimBuffer:
        return vk::DescriptorType::eStorageTexelBuffer;
        break;
    default:
        WHEELS_ASSERT(!"Unimplemented image dimensionality");
        return vk::DescriptorType::eSampler;
    }
}

Array<DescriptorSetMetadata> &getSetMetadatas(
    const Decorations &decorations,
    HashMap<uint32_t, Array<DescriptorSetMetadata>> &metadatas)
{
    WHEELS_ASSERT(decorations.descriptorSet != sUninitialized);

    Array<DescriptorSetMetadata> *setMetadatas =
        metadatas.find(decorations.descriptorSet);
    WHEELS_ASSERT(setMetadatas != nullptr);

    return *setMetadatas;
}

const SpvResult &getType(
    const SpvVariable &variable, Span<const SpvResult> results)
{
    const SpvResult &typePtrResult = results[variable.typeId];
    WHEELS_ASSERT(typePtrResult.type.has_value());
    WHEELS_ASSERT(std::holds_alternative<SpvPointer>(*typePtrResult.type));
    const SpvPointer &typePtr = std::get<SpvPointer>(*typePtrResult.type);

    const SpvResult &typeResult = results[typePtr.typeId];
    WHEELS_ASSERT(typeResult.type.has_value());

    return typeResult;
}

vk::DescriptorType intoArrayDescriptorType(const SpvResult &typeResult)
{
    WHEELS_ASSERT(typeResult.type.has_value());

    if (std::holds_alternative<SpvSampler>(*typeResult.type))
        return vk::DescriptorType::eSampler;

    if (std::holds_alternative<SpvSampledImage>(*typeResult.type))
        return vk::DescriptorType::eCombinedImageSampler;

    if (const SpvImage *iimage = std::get_if<SpvImage>(&*typeResult.type);
        iimage != nullptr)
        return imageDescriptorType(*iimage);

    WHEELS_ASSERT(!"Unimplemented variant");
    return vk::DescriptorType::eSampler;
}

bool isDynamicStorageBuffer(
    const SpvVariable &variable, Span<const SpvResult> results)
{
    const SpvResult &type = getType(variable, results);
    if (type.name == nullptr)
        return false;

    // Let's just label dynamic SBs in the shader buffer type as that doesn't
    // bleed into the accessing shader code and still gets us the 'correct'
    // reflection every time. This means we can't use the binding as both SSB
    // and dynamic SSB in different passes but let's not complicate the
    // interface until we have to.
    const char *postfix = "DSB";
    const size_t postfixLength = strlen(postfix);

    const size_t typeNameLen = strlen(type.name);
    bool isDynamic = false;
    if (typeNameLen > postfixLength)
        isDynamic = strncmp(
                        type.name + typeNameLen - postfixLength, postfix,
                        postfixLength) == 0;

    return isDynamic;
}

void fillMetadata(
    String &&name, const Decorations &decorations, const SpvVariable &variable,
    Span<const SpvResult> results,
    HashMap<uint32_t, Array<DescriptorSetMetadata>> &metadatas)
{
    // TODO: Generalize the common parts, pull out case noise into
    // helpers
    bool fill = false;

    vk::DescriptorType descriptorType = vk::DescriptorType::eSampler;
    uint32_t descriptorCount = 1;
    switch (variable.storageClass)
    {
    case spv::StorageClassStorageBuffer:
    {
        fill = true;

        descriptorType = isDynamicStorageBuffer(variable, results)
                             ? vk::DescriptorType::eStorageBufferDynamic
                             : vk::DescriptorType::eStorageBuffer;

        const SpvResult &typeResult = getType(variable, results);
        if (std::holds_alternative<SpvRuntimeArray>(*typeResult.type))
            descriptorCount = 0;
        else // Struct is the default count 1
             // This might fire when a runtime array bind is declared but not
             // actually used
            WHEELS_ASSERT(std::holds_alternative<SpvStruct>(*typeResult.type));
    }
    break;
    case spv::StorageClassUniform:
    {
        fill = true;
        descriptorType = vk::DescriptorType::eUniformBuffer;

        const SpvResult &typeResult = getType(variable, results);
        WHEELS_ASSERT(std::holds_alternative<SpvStruct>(*typeResult.type));
    }
    break;
    case spv::StorageClassUniformConstant:
    {
        fill = true;

        const SpvResult &typeResult = getType(variable, results);
        if (std::holds_alternative<SpvSampler>(*typeResult.type))
            descriptorType = vk::DescriptorType::eSampler;
        else if (std::holds_alternative<SpvSampledImage>(*typeResult.type))
            descriptorType = vk::DescriptorType::eCombinedImageSampler;
        else if (const SpvImage *image =
                     std::get_if<SpvImage>(&*typeResult.type);
                 image != nullptr)
            descriptorType = imageDescriptorType(*image);
        else if (const SpvArray *array =
                     std::get_if<SpvArray>(&*typeResult.type);
                 array != nullptr)
        {
            descriptorType =
                intoArrayDescriptorType(results[array->elementTypeId]);
            descriptorCount = array->length;
        }
        else if (const SpvRuntimeArray *runtimeArray =
                     std::get_if<SpvRuntimeArray>(&*typeResult.type);
                 runtimeArray != nullptr)
        {
            descriptorType =
                intoArrayDescriptorType(results[runtimeArray->elementTypeId]);
            descriptorCount = 0;
        }
        else if (std::holds_alternative<SpvAccelerationStructure>(
                     *typeResult.type))
        {
            descriptorType = vk::DescriptorType::eAccelerationStructureKHR;
        }
        else
            WHEELS_ASSERT(!"Unimplemented variant");
    }
    break;
    default:
        break;
    }

    if (fill)
    {
        WHEELS_ASSERT(decorations.binding != sUninitialized);

        Array<DescriptorSetMetadata> &setMetadatas =
            getSetMetadatas(decorations, metadatas);

        setMetadatas.push_back(DescriptorSetMetadata{
            .name = WHEELS_MOV(name),
            .binding = decorations.binding,
            .descriptorType = descriptorType,
            .descriptorCount = descriptorCount,
        });
    }
}

HashMap<uint32_t, Array<DescriptorSetMetadata>> fillDescriptorSetMetadatas(
    ScopedScratch scopeAlloc, Span<const SpvResult> results)
{
    // Get counts first so we can allocate return memory exactly
    HashMap<uint32_t, uint32_t> descriptorSetBindingCounts{scopeAlloc, 16};
    for (const SpvResult &result : results)
    {
        if (result.decorations.descriptorSet != sUninitialized)
        {
            if (descriptorSetBindingCounts.contains(
                    result.decorations.descriptorSet))
            {
                uint32_t &count = *descriptorSetBindingCounts.find(
                    result.decorations.descriptorSet);
                count++;
            }
            else
            {
                descriptorSetBindingCounts.insert_or_assign(
                    result.decorations.descriptorSet, 1u);
            }
        }
    }

    HashMap<uint32_t, Array<DescriptorSetMetadata>> ret{
        gAllocators.general, descriptorSetBindingCounts.size() * 2};
    for (const auto &iter : descriptorSetBindingCounts)
        ret.insert_or_assign(
            *iter.first,
            Array<DescriptorSetMetadata>{gAllocators.general, *iter.second});

    // Fill the metadata
    for (const SpvResult &result : results)
    {
        // All descriptor bindings should have a name
        if (result.name == nullptr)
            continue;

        // Is this use case why std::get_if takes in a pointer instead of a
        // reference?
        const SpvType *typePtr =
            result.type.has_value() ? &*result.type : nullptr;
        if (const SpvVariable *variable = std::get_if<SpvVariable>(typePtr);
            variable != nullptr)
        {
            fillMetadata(
                String{gAllocators.general, result.name}, result.decorations,
                *variable, results, ret);
        }
    }

    for (const auto &index_metadatas : ret)
    {
        // Make sure metadatas are sorted by binding indices as we depend on it
        // when generating writes
        Array<DescriptorSetMetadata> &metadatas = *index_metadatas.second;
        std::sort(
            metadatas.begin(), metadatas.end(),
            [](const DescriptorSetMetadata &a, const DescriptorSetMetadata &b)
            { return a.binding < b.binding; });

        // Get rid of aliased storage buffer bindings so that we just have the
        // one to generate writes for
        for (size_t i = 1; i < metadatas.size(); ++i)
        {
            const DescriptorSetMetadata &current = metadatas[i];
            DescriptorSetMetadata &previous = metadatas[i - 1];
            if (current.binding == previous.binding)
            {
                WHEELS_ASSERT(
                    current.descriptorType == previous.descriptorType);
                WHEELS_ASSERT(
                    current.descriptorType ==
                        vk::DescriptorType::eStorageBuffer ||
                    current.descriptorType ==
                        vk::DescriptorType::eStorageBufferDynamic);

                // Concat aliased names so the aliasing is clear when generating
                // layouts or binds
                previous.name.push_back('|');
                previous.name.extend(current.name);

                metadatas.erase(i);
                i--;
            }
        }
    }

    return ret;
}

Array<vk::SpecializationMapEntry> fillSpecializationMap(
    Span<const SpvResult> results)
{
    Array<vk::SpecializationMapEntry> ret{gAllocators.general};
    for (const SpvResult &result : results)
    {
        if (!result.type.has_value())
            continue;

        if (const SpvSpecializationConstant *constantPtr =
                std::get_if<SpvSpecializationConstant>(&*result.type);
            constantPtr != nullptr)
        {
            if (constantPtr->id >= ret.size())
                ret.resize(
                    constantPtr->id + 1, vk::SpecializationMapEntry{
                                             .constantID = sUninitialized,
                                             .offset = sUninitialized,
                                             .size = sUninitialized,
                                         });

            ret[constantPtr->id] = vk::SpecializationMapEntry{
                .constantID = constantPtr->id,
                .offset = 0,
                .size = constantPtr->size,
            };
        }
    }

    const uint32_t constantCount = asserted_cast<uint32_t>(ret.size());
    for (const vk::SpecializationMapEntry &entry : ret)
    {
        if (entry.constantID == sUninitialized)
            throw std::runtime_error(
                "Specialization constants that are not results from ops have "
                "to populate the indices from 0 without gaps.");
    }

    for (uint32_t i = 1; i < constantCount; ++i)
    {
        const vk::SpecializationMapEntry &previousEntry = ret[i - 1];
        vk::SpecializationMapEntry &currentEntry = ret[i];
        currentEntry.offset = asserted_cast<uint32_t>(
            asserted_cast<size_t>(previousEntry.offset) + previousEntry.size);
        WHEELS_ASSERT(
            currentEntry.offset % currentEntry.size == 0 &&
            "Inferred specialization constant offset might not satisfy "
            "alignment");
    }

    return ret;
}

} // namespace

ShaderReflection::ShaderReflection(ShaderReflection &&other) noexcept
: m_initialized{other.m_initialized}
, m_pushConstantsBytesize{other.m_pushConstantsBytesize}
, m_descriptorSetMetadatas{WHEELS_MOV(other.m_descriptorSetMetadatas)}
, m_sourceFiles{WHEELS_MOV(other.m_sourceFiles)}
, m_specializationMapEntries{WHEELS_MOV(other.m_specializationMapEntries)}
, m_specializationConstantsByteSize{other.m_specializationConstantsByteSize}
{
}

ShaderReflection &ShaderReflection::operator=(ShaderReflection &&other) noexcept
{
    if (this != &other)
    {
        WHEELS_ASSERT(!m_initialized);

        m_initialized = other.m_initialized;
        m_pushConstantsBytesize = other.m_pushConstantsBytesize;
        m_descriptorSetMetadatas = WHEELS_MOV(other.m_descriptorSetMetadatas);
        m_sourceFiles = WHEELS_MOV(other.m_sourceFiles);
        m_specializationMapEntries =
            WHEELS_MOV(other.m_specializationMapEntries);
        m_specializationConstantsByteSize =
            other.m_specializationConstantsByteSize;
    }
    return *this;
}

void ShaderReflection::init(
    ScopedScratch scopeAlloc, Span<const uint32_t> spvWords,
    const wheels::HashSet<std::filesystem::path> &sourceFiles)
{
    WHEELS_ASSERT(!m_initialized);

    for (const std::filesystem::path &include : sourceFiles)
        m_sourceFiles.insert(include);

    const uint32_t spvMagic = 0x0723'0203;
    if (spvWords[0] != spvMagic)
        throw std::runtime_error(
            "Tried to read reflection from invalid SPIR-V words");

    // bytes 0 | major | minor | 0, 0x0001'0300 is 1.3
    // const uint32_t version = words[1];
    // const uint32_t generatorMagic = spvWords[2];
    const uint32_t idBound = spvWords[3];
    // const uint32_t schema = spvWords[4];

    Array<SpvResult> results{scopeAlloc};
    results.resize(idBound);

    uint32_t pushConstantMetadataId = sUninitialized;

    // Run in two passes because type definitons come after decorations.
    // Data relations are simpler this way.
    firstPass(scopeAlloc, spvWords, results.mut_span(), pushConstantMetadataId);
    secondPass(spvWords, results.mut_span());

    if (pushConstantMetadataId != sUninitialized)
        m_pushConstantsBytesize =
            getPushConstantsBytesize(results, pushConstantMetadataId);

    m_descriptorSetMetadatas =
        fillDescriptorSetMetadatas(scopeAlloc.child_scope(), results);

    m_specializationMapEntries = fillSpecializationMap(results);
    if (!m_specializationMapEntries.empty())
    {
        const vk::SpecializationMapEntry &lastConstant =
            m_specializationMapEntries.back();
        m_specializationConstantsByteSize =
            lastConstant.offset + lastConstant.size;
    }

    m_initialized = true;
}

uint32_t ShaderReflection::pushConstantsBytesize() const
{
    WHEELS_ASSERT(m_initialized);

    return m_pushConstantsBytesize;
}

HashMap<uint32_t, Array<DescriptorSetMetadata>> const &ShaderReflection::
    descriptorSetMetadatas() const
{
    WHEELS_ASSERT(m_initialized);

    return m_descriptorSetMetadatas;
}

uint32_t ShaderReflection::specializationConstantsByteSize() const
{
    return m_specializationConstantsByteSize;
}

wheels::Span<const vk::SpecializationMapEntry> ShaderReflection::
    specializationMapEntries() const
{
    return m_specializationMapEntries;
}

const HashSet<std::filesystem::path> &ShaderReflection::sourceFiles() const
{
    WHEELS_ASSERT(m_initialized);

    return m_sourceFiles;
}

bool ShaderReflection::affected(
    const HashSet<std::filesystem::path> &changedFiles) const
{
    WHEELS_ASSERT(m_initialized);

    bool found = false;
    for (const std::filesystem::path &file : changedFiles)
    {
        if (m_sourceFiles.contains(file))
        {
            found = true;
            break;
        }
    }
    return found;
}

vk::DescriptorSetLayout ShaderReflection::createDescriptorSetLayout(
    ScopedScratch scopeAlloc, uint32_t descriptorSet,
    vk::ShaderStageFlags stageFlags, wheels::Span<const uint32_t> dynamicCounts,
    wheels::Span<const vk::DescriptorBindingFlags> bindingFlags) const
{
    WHEELS_ASSERT(m_initialized);

    const Array<DescriptorSetMetadata> *metadatas =
        m_descriptorSetMetadatas.find(descriptorSet);
    WHEELS_ASSERT(metadatas != nullptr);

    Array<vk::DescriptorSetLayoutBinding> layoutBindings{
        scopeAlloc, metadatas->size()};

    size_t currentDynamicCount = 0;
    for (const DescriptorSetMetadata &metadata : *metadatas)
    {
        const uint32_t descriptorCount =
            (metadata.descriptorCount > 0)
                ? metadata.descriptorCount
                : dynamicCounts[currentDynamicCount++];

        layoutBindings.push_back(vk::DescriptorSetLayoutBinding{
            .binding = metadata.binding,
            .descriptorType = metadata.descriptorType,
            .descriptorCount = descriptorCount,
            .stageFlags = stageFlags,
        });
    }
    WHEELS_ASSERT(
        currentDynamicCount == dynamicCounts.size() &&
        "Extra dynamic counts given");

    if (bindingFlags.empty())
        return gDevice.logical().createDescriptorSetLayout(
            vk::DescriptorSetLayoutCreateInfo{
                .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
                .pBindings = layoutBindings.data(),
            });

    WHEELS_ASSERT(
        bindingFlags.size() == layoutBindings.size() &&
        "Binding flag count has to match binding count");
    const vk::StructureChain<
        vk::DescriptorSetLayoutCreateInfo,
        vk::DescriptorSetLayoutBindingFlagsCreateInfo>
        layoutChain{
            vk::DescriptorSetLayoutCreateInfo{
                .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
                .pBindings = layoutBindings.data(),
            },
            vk::DescriptorSetLayoutBindingFlagsCreateInfo{
                .bindingCount = asserted_cast<uint32_t>(bindingFlags.size()),
                .pBindingFlags = bindingFlags.data(),
            }};
    return gDevice.logical().createDescriptorSetLayout(
        layoutChain.get<vk::DescriptorSetLayoutCreateInfo>());
}

wheels::Array<vk::WriteDescriptorSet> ShaderReflection::
    generateDescriptorWrites(
        Allocator &alloc, uint32_t descriptorSetIndex,
        vk::DescriptorSet descriptorSetHandle,
        wheels::Span<const DescriptorInfo> descriptorInfos) const
{
    WHEELS_ASSERT(m_initialized);

    const wheels::Array<DescriptorSetMetadata> *metadatas =
        m_descriptorSetMetadatas.find(descriptorSetIndex);
    WHEELS_ASSERT(metadatas != nullptr);
    // false positive, custom assert above
    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    WHEELS_ASSERT(
        metadatas->size() == descriptorInfos.size() &&
        "All binds should have a descriptor info. Arrays that are left unbound "
        "should have an empty span.");

    wheels::Array<vk::WriteDescriptorSet> descriptorWrites{
        alloc, descriptorInfos.size()};

    const size_t infoCount = descriptorInfos.size();
    for (size_t i = 0; i < infoCount; ++i)
    {
        const DescriptorInfo &descriptorInfo = descriptorInfos[i];

        const vk::DescriptorImageInfo *pImageInfo =
            std::get_if<vk::DescriptorImageInfo>(&descriptorInfo);
        const vk::DescriptorBufferInfo *pBufferInfo =
            std::get_if<vk::DescriptorBufferInfo>(&descriptorInfo);
        const vk::BufferView *pTexelBufferView =
            std::get_if<vk::BufferView>(&descriptorInfo);
        // TODO:
        // Refactor this so that single image is also a span? How are the
        // ergonomics?
        const wheels::Span<const vk::DescriptorImageInfo> *pImageInfoSpan =
            std::get_if<wheels::Span<const vk::DescriptorImageInfo>>(
                &descriptorInfo);
        const wheels::Span<const vk::DescriptorBufferInfo> *pBufferInfoSpan =
            std::get_if<wheels::Span<const vk::DescriptorBufferInfo>>(
                &descriptorInfo);

        uint32_t descriptorCount = 1;

        if (pImageInfoSpan != nullptr)
        {
            pImageInfo = pImageInfoSpan->data();
            descriptorCount = asserted_cast<uint32_t>(pImageInfoSpan->size());
        }
        else if (pBufferInfoSpan != nullptr)
        {
            pBufferInfo = pBufferInfoSpan->data();
            descriptorCount = asserted_cast<uint32_t>(pBufferInfoSpan->size());
        }

        // Zero is expected when descriptors are left unbound explicitly
        if (descriptorCount > 0)
        {
            const DescriptorSetMetadata &metadata = (*metadatas)[i];
            descriptorWrites.push_back(vk::WriteDescriptorSet{
                .dstSet = descriptorSetHandle,
                .dstBinding = metadata.binding,
                .descriptorCount = descriptorCount,
                .descriptorType = metadata.descriptorType,
                .pImageInfo = pImageInfo,
                .pBufferInfo = pBufferInfo,
                .pTexelBufferView = pTexelBufferView,
            });
        }
    }

    return descriptorWrites;
}
