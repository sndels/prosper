#include "ShaderReflection.hpp"

#include <spirv.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/optional.hpp>

#include <cstring>
#include <stdexcept>
#include <variant>

using namespace wheels;

namespace
{

struct SpvInt;
struct SpvFloat;
struct SpvVector;
struct SpvMatrix;
struct SpvImage;
struct SpvSampler;
struct SpvArray;
struct SpvRuntimeArray;
struct SpvStruct;
struct SpvPointer;
struct SpvAccelerationStructure;
struct SpvConstantU32;
struct SpvVariable;

// SpvVariable is not a really a type-type, but it is a type of result
using SpvType = Optional<std::variant<
    SpvInt, SpvFloat, SpvVector, SpvMatrix, SpvImage, SpvSampler,
    SpvRuntimeArray, SpvArray, SpvStruct, SpvPointer, SpvAccelerationStructure,
    SpvConstantU32, SpvVariable>>;

// From https://en.cppreference.com/w/cpp/utility/variant/visit
template <class... Ts> struct overloaded : Ts...
{
    using Ts::operator()...;
};
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

const uint32_t sUninitialized = 0xFFFFFFFF;

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

struct Decorations
{
    uint32_t descriptorSet{sUninitialized};
    uint32_t binding{sUninitialized};
};

// Only valid until the bytecode is freed
struct SpvResult
{
    const char *name{nullptr};
    SpvType type;
    Decorations decorations;
};

const size_t firstOpOffset = 5;

void firstPass(
    Allocator &alloc, const uint32_t *words, size_t wordCount,
    Array<SpvResult> &results, uint32_t &pushConstantMetadataId)
{
    // Collect names and types
    size_t opFirstWord = firstOpOffset;
    while (opFirstWord < wordCount)
    {
        const uint16_t opWordCount =
            static_cast<uint16_t>(words[opFirstWord] >> 16);
        const uint16_t op = static_cast<uint16_t>(words[opFirstWord] & 0xFFFF);
        const uint32_t *args = &words[opFirstWord + 1];

        switch (op)
        {
        case spv::OpName:
        {
            const uint32_t result = args[0];
            const char *name = reinterpret_cast<const char *>(&args[1]);

            results[result].name = name;
        }
        break;
        case spv::OpTypeInt:
        {
            const uint32_t result = args[0];
            const uint32_t width = args[1];
            const uint32_t signedness = args[2];
            assert(signedness == 0 || signedness == 1);

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
            assert(lengthResult.type.has_value());
            const SpvConstantU32 *length =
                std::get_if<SpvConstantU32>(&*lengthResult.type);
            assert(length != nullptr);
            assert(length->value != sUninitialized);

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
                const SpvType &type = results[typeId].type;
                // Accessors into PC struct members are also this storage class,
                // so let's just pick the struct
                if (type.has_value() &&
                    std::holds_alternative<SpvStruct>(*type))
                {
                    // This probably fires if we have structs within the push
                    // constants struct, if that's even possible
                    assert(
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
            assert(type.type.has_value());

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

void secondPass(
    const uint32_t *words, size_t wordCount, Array<SpvResult> &results)
{
    // Collect decorations
    size_t opFirstWord = firstOpOffset;
    while (opFirstWord < wordCount)
    {
        const uint16_t opWordCount =
            static_cast<uint16_t>(words[opFirstWord] >> 16);
        const uint16_t op = static_cast<uint16_t>(words[opFirstWord] & 0xFFFF);
        const uint32_t *args = &words[opFirstWord + 1];

        switch (op)
        {
        case spv::OpDecorate:
        {
            const uint32_t resultId = args[0];
            const uint32_t decoration = static_cast<spv::Decoration>(args[1]);

            SpvResult &result = results[resultId];

            switch (decoration)
            {
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
                assert(spvStruct != nullptr);

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
    const SpvType &type, const MemberDecorations &memberDecorations,
    const Array<SpvResult> &results)
{
    assert(
        type.has_value() && "Unimplemented member type, probably OpTypeArray");

    return std::visit(
        overloaded{
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
                assert(memberDecorations.matrixStride != sUninitialized);
                return memberDecorations.matrixStride * v.columnCount;
            },
            [&results](const SpvStruct &v) -> uint32_t
            {
                const uint32_t lastMemberId = v.memberTypeIds.back();
                const MemberDecorations &lastMemberDecorations =
                    v.memberDecorations.back();

                const SpvResult &lastMemberResult = results[lastMemberId];

                const uint32_t lastMemberBytesize = memberBytesize(
                    lastMemberResult.type, MemberDecorations{}, results);

                assert(lastMemberDecorations.offset != sUninitialized);
                const uint32_t bytesize =
                    lastMemberDecorations.offset + lastMemberBytesize;

                return bytesize;
            },
            [](const auto &) -> uint32_t
            {
                assert(!"Unimplemented");
                return 0;
            }},
        *type);
}

uint32_t getPushConstantsBytesize(
    const Array<SpvResult> &results, uint32_t metadataId)
{
    const SpvResult &pcResult = results[metadataId];

    return memberBytesize(pcResult.type, MemberDecorations{}, results);
}

HashMap<uint32_t, Array<DescriptorSetMetadata>> fillDescriptorSetMetadatas(
    ScopedScratch scopeAlloc, Allocator &alloc, const Array<SpvResult> &results)
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
        alloc, descriptorSetBindingCounts.size() * 2};
    for (const auto &iter : descriptorSetBindingCounts)
        ret.insert_or_assign(
            *iter.first, Array<DescriptorSetMetadata>{alloc, *iter.second});

    // Fill the metadata
    for (const SpvResult &result : results)
    {
        if (result.type.has_value())
        {
            if (const SpvVariable *variable =
                    std::get_if<SpvVariable>(&*result.type);
                variable != nullptr)
            {
                // TODO: Generalize the common parts
                switch (variable->storageClass)
                {
                case spv::StorageClassStorageBuffer:
                {
                    const uint32_t descriptorSet =
                        result.decorations.descriptorSet;
                    assert(descriptorSet != sUninitialized);
                    const uint32_t binding = result.decorations.binding;
                    assert(binding != sUninitialized);

                    Array<DescriptorSetMetadata> *setMetadatas =
                        ret.find(descriptorSet);
                    assert(setMetadatas != nullptr);

                    const SpvResult &typePtrResult = results[variable->typeId];
                    assert(typePtrResult.type.has_value());
                    assert(std::holds_alternative<SpvPointer>(
                        *typePtrResult.type));
                    const SpvPointer &typePtr =
                        std::get<SpvPointer>(*typePtrResult.type);

                    const SpvResult &typeResult = results[typePtr.typeId];
                    assert(typeResult.type.has_value());

                    uint32_t descriptorCount = 0;
                    if (std::holds_alternative<SpvStruct>(*typeResult.type))
                        descriptorCount = 1;
                    else // Assert the initialized 0 is correct
                        assert(std::holds_alternative<SpvRuntimeArray>(
                            *typeResult.type));

                    setMetadatas->push_back(DescriptorSetMetadata{
                        .name = String{alloc, result.name},
                        .binding = binding,
                        .descriptorType = vk::DescriptorType::eStorageBuffer,
                        .descriptorCount = descriptorCount,
                    });
                }
                break;
                case spv::StorageClassUniform:
                {
                    const uint32_t descriptorSet =
                        result.decorations.descriptorSet;
                    assert(descriptorSet != sUninitialized);
                    const uint32_t binding = result.decorations.binding;
                    assert(binding != sUninitialized);

                    Array<DescriptorSetMetadata> *setMetadatas =
                        ret.find(descriptorSet);
                    assert(setMetadatas != nullptr);

                    const SpvResult &typePtrResult = results[variable->typeId];
                    assert(typePtrResult.type.has_value());
                    assert(std::holds_alternative<SpvPointer>(
                        *typePtrResult.type));
                    const SpvPointer &typePtr =
                        std::get<SpvPointer>(*typePtrResult.type);

                    const SpvResult &typeResult = results[typePtr.typeId];
                    assert(typeResult.type.has_value());
                    assert(std::holds_alternative<SpvStruct>(*typeResult.type));

                    setMetadatas->push_back(DescriptorSetMetadata{
                        .name = String{alloc, result.name},
                        .binding = binding,
                        .descriptorType = vk::DescriptorType::eUniformBuffer,
                        .descriptorCount = 1,
                    });
                }
                break;
                default:
                    break;
                }
            }
        }
    }

    return ret;
}

} // namespace

ShaderReflection::ShaderReflection(
    ScopedScratch scopeAlloc, Allocator &alloc, Span<const uint32_t> spvWords)
: _descriptorSetMetadatas{alloc}
{
    const uint32_t *words = spvWords.data();
    const size_t wordCount = spvWords.size();

    const uint32_t spvMagic = 0x07230203;
    if (words[0] != spvMagic)
        throw std::runtime_error(
            "Tried to read reflection from invalid SPIR-V words");

    // bytes 0 | major | minor | 0, 0x00010300 is 1.3
    // const uint32_t version = words[1];
    // const uint32_t generatorMagic = words[2];
    const uint32_t idBound = words[3];
    // const uint32_t schema = words[4];

    Array<SpvResult> results{alloc};
    results.resize(idBound);

    uint32_t pushConstantMetadataId = sUninitialized;

    // Run in two passes because type definitons come after decorations.
    // Data relations are simpler this way.
    firstPass(scopeAlloc, words, wordCount, results, pushConstantMetadataId);
    secondPass(words, wordCount, results);

    if (pushConstantMetadataId != sUninitialized)
        _pushConstantsBytesize =
            getPushConstantsBytesize(results, pushConstantMetadataId);

    _descriptorSetMetadatas =
        fillDescriptorSetMetadatas(scopeAlloc.child_scope(), alloc, results);
}

uint32_t ShaderReflection::pushConstantsBytesize() const
{
    return _pushConstantsBytesize;
}

HashMap<uint32_t, Array<DescriptorSetMetadata>> const &ShaderReflection::
    descriptorSetMetadatas() const
{
    return _descriptorSetMetadatas;
}
