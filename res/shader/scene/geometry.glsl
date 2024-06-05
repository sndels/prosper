#ifndef SCENE_GEOMETRY_GLSL
#define SCENE_GEOMETRY_GLSL

#include "vertex.glsl"

struct MeshBuffer
{
    uint index;
    uint offset;
};

struct GeometryMetadata
{
    uint bufferIndex;
    // These offsets are into the geometry data buffers. Most are for U32/F32
    // and an offset of 0xFFFFFFFF signals an unused attribute.
    // This addresses U16 if short indices are in use.
    uint indicesOffset;
    uint positionsOffset;
    uint normalsOffset;
    uint tangentsOffset;
    uint texCoord0sOffset;
    uint meshletsOffset;
    uint meshletBoundsOffset;
    // This addresses U16 if short indices are in use.
    uint meshletVerticesOffset;
    // This addresses U8.
    uint meshletTrianglesByteOffset;
    uint usesShortIndices;
};
layout(std430, set = GEOMETRY_SET, binding = 0) readonly buffer
    GeometryMetadatas
{
    GeometryMetadata data[];
}
geometryMetadatas;

layout(std430, set = GEOMETRY_SET, binding = 1) readonly buffer MeshletCounts
{
    uint data[];
}
meshletCounts;

// Aliased binds of the same SSBOs
layout(std430, set = GEOMETRY_SET, binding = 2) readonly buffer
    GeometryBuffersU32
{
    uint data[];
}
geometryBuffersU32[];
layout(std430, set = GEOMETRY_SET, binding = 2) readonly buffer
    GeometryBuffersU16
{
    uint16_t data[];
}
geometryBuffersU16[];
layout(std430, set = GEOMETRY_SET, binding = 2) readonly buffer
    GeometryBuffersU8
{
    uint8_t data[];
}
geometryBuffersU8[];
layout(std430, set = GEOMETRY_SET, binding = 2) readonly buffer
    GeometryBuffersF32
{
    float data[];
}
geometryBuffersF32[];

#define GET_GEOMETRY_BUFFER_U32(index) geometryBuffersU32[nonuniformEXT(index)]
#define GET_GEOMETRY_BUFFER_U16(index) geometryBuffersU16[nonuniformEXT(index)]
#define GET_GEOMETRY_BUFFER_U8(index) geometryBuffersU8[nonuniformEXT(index)]
#define GET_GEOMETRY_BUFFER_F32(index) geometryBuffersF32[nonuniformEXT(index)]

uint loadIndex(
    uint bufferIndex, uint bufferOffset, uint index, uint usesShortIndices)
{
    if (usesShortIndices == 1)
        return uint(
            GET_GEOMETRY_BUFFER_U16(bufferIndex).data[bufferOffset + index]);
    else
        return GET_GEOMETRY_BUFFER_U32(bufferIndex).data[bufferOffset + index];
}

float loadFloat(uint bufferIndex, uint bufferOffset, uint index)
{
    return GET_GEOMETRY_BUFFER_F32(bufferIndex).data[bufferOffset + index];
}

uint loadU8(uint bufferIndex, uint bufferOffset, uint index)
{
    return uint(GET_GEOMETRY_BUFFER_U8(bufferIndex).data[bufferOffset + index]);
}

vec2 loadR16G16(uint bufferIndex, uint bufferOffset, uint index)
{
    if (bufferOffset == 0xFFFFFFFF)
        return vec2(0);

    uint packed =
        GET_GEOMETRY_BUFFER_U32(bufferIndex).data[bufferOffset + index];

    return unpackHalf2x16(packed);
}

vec3 loadR16G16B16A16(uint bufferIndex, uint bufferOffset, uint index)
{
    if (bufferOffset == 0xFFFFFFFF)
        return vec3(0);

    uvec2 packed = uvec2(
        GET_GEOMETRY_BUFFER_U32(bufferIndex).data[bufferOffset + index * 2],
        GET_GEOMETRY_BUFFER_U32(bufferIndex)
            .data[bufferOffset + index * 2 + 1]);

    return vec3(unpackHalf2x16(packed.x), unpackHalf2x16(packed.y));
}

vec3 unpackSnormR10G10B10(uint packed)
{
    // Leverage sign extension in GLSL right shift to unpack the components
    // https://www.gamedev.net/forums/topic/696946-normalized-unsigned-integers-vs-floats-as-vertex-data/5379938/
    ivec3 signExtended = ivec3(packed << 22, packed << 12, packed << 2) >> 22;

    // 3.10.1. Conversion From Normalized Fixed-Point to Floating-Point
    return normalize(max(vec3(signExtended) / 511., -1));
}

vec3 loadR10G10B10Snorm(uint bufferIndex, uint bufferOffset, uint index)
{
    if (bufferOffset == 0xFFFFFFFF)
        return vec3(0);

    uint packed =
        GET_GEOMETRY_BUFFER_U32(bufferIndex).data[bufferOffset + index];

    return unpackSnormR10G10B10(packed);
}

vec4 loadTangentWithSign(uint bufferIndex, uint bufferOffset, uint index)
{
    if (bufferOffset == 0xFFFFFFFF)
        return vec4(0);

    uint packed =
        GET_GEOMETRY_BUFFER_U32(bufferIndex).data[bufferOffset + index];

    // Leverage sign extension in GLSL right shift to unpack the sign
    // https://www.gamedev.net/forums/topic/696946-normalized-unsigned-integers-vs-floats-as-vertex-data/5379938/
    return vec4(unpackSnormR10G10B10(packed), float(int(packed) >> 30));
}

struct MeshletInfo
{
    // This is an offset of full indices from the beginning of the meshlet
    // vertices buffer so it works for both U32 and U16.
    uint vertexOffset;
    // This addresses U8.
    uint triangleByteOffset;
    uint vertexCount;
    uint triangleCount;
};

MeshletInfo loadMeshletInfo(GeometryMetadata metadata, uint index)
{
    MeshletInfo ret;
    ret.vertexOffset = GET_GEOMETRY_BUFFER_U32(metadata.bufferIndex)
                           .data[metadata.meshletsOffset + index * 4];
    ret.triangleByteOffset = GET_GEOMETRY_BUFFER_U32(metadata.bufferIndex)
                                 .data[metadata.meshletsOffset + index * 4 + 1];
    ret.vertexCount = GET_GEOMETRY_BUFFER_U32(metadata.bufferIndex)
                          .data[metadata.meshletsOffset + index * 4 + 2];
    ret.triangleCount = GET_GEOMETRY_BUFFER_U32(metadata.bufferIndex)
                            .data[metadata.meshletsOffset + index * 4 + 3];

    return ret;
}

struct MeshletBounds
{
    // Bounding sphere
    vec3 center;
    float radius;
    // Normal cone
    vec3 coneAxis;
    float coneCutoff;
};
#define MESHLET_BOUNDS_F32_COUNT 11

MeshletBounds loadMeshletBounds(GeometryMetadata metadata, uint index)
{
    MeshletBounds ret;
    ret.center = vec3(
        loadFloat(
            metadata.bufferIndex, metadata.meshletBoundsOffset, index * 5),
        loadFloat(
            metadata.bufferIndex, metadata.meshletBoundsOffset, index * 5 + 1),
        loadFloat(
            metadata.bufferIndex, metadata.meshletBoundsOffset, index * 5 + 2));
    ret.radius = loadFloat(
        metadata.bufferIndex, metadata.meshletBoundsOffset, index * 5 + 3);

    // int<->uint is actually a proper bitcast in GLSL
    int packedCone =
        int(GET_GEOMETRY_BUFFER_U32(metadata.bufferIndex)
                .data[metadata.meshletBoundsOffset + index * 5 + 4]);
    // This shift dance leverages GLSL right shift sign extension to unpack the
    // signed ingeter components
    ret.coneAxis =
        vec3(ivec3(packedCone << 24, packedCone << 16, packedCone << 8) >> 24) /
        127.;
    ret.coneCutoff = (packedCone >> 24) / 127.;

    return ret;
}

uint loadMeshletVertexIndex(
    GeometryMetadata metadata, MeshletInfo info, uint index)
{
    return loadIndex(
        metadata.bufferIndex,
        metadata.meshletVerticesOffset + info.vertexOffset, index,
        metadata.usesShortIndices);
}

uvec3 loadMeshletTriangle(
    GeometryMetadata metadata, MeshletInfo info, uint index)
{
    return uvec3(
        loadU8(
            metadata.bufferIndex,
            metadata.meshletTrianglesByteOffset + info.triangleByteOffset,
            index * 3 + 0),
        loadU8(
            metadata.bufferIndex,
            metadata.meshletTrianglesByteOffset + info.triangleByteOffset,
            index * 3 + 1),
        loadU8(
            metadata.bufferIndex,
            metadata.meshletTrianglesByteOffset + info.triangleByteOffset,
            index * 3 + 2));
}

Vertex loadVertex(GeometryMetadata metadata, uint index)
{
    Vertex ret;

    ret.Position =
        loadR16G16B16A16(metadata.bufferIndex, metadata.positionsOffset, index)
            .xyz;
    ret.Normal =
        loadR10G10B10Snorm(metadata.bufferIndex, metadata.normalsOffset, index);
    ret.Tangent = loadTangentWithSign(
        metadata.bufferIndex, metadata.tangentsOffset, index);
    ret.TexCoord0 =
        loadR16G16(metadata.bufferIndex, metadata.texCoord0sOffset, index);

    return ret;
}

Vertex loadVertexThroughIndexBuffer(GeometryMetadata metadata, uint index)
{
    uint vertexIndex = loadIndex(
        metadata.bufferIndex, metadata.indicesOffset, index,
        metadata.usesShortIndices);

    return loadVertex(metadata, vertexIndex);
}

vec2 loadUV(uint meshID, uint index)
{
    GeometryMetadata metadata = geometryMetadatas.data[meshID];

    uint vertexIndex = loadIndex(
        metadata.bufferIndex, metadata.indicesOffset, index,
        metadata.usesShortIndices);

    return loadR16G16(
        metadata.bufferIndex, metadata.texCoord0sOffset, vertexIndex);
}

vec2 baryInterpolate(vec2 v0, vec2 v1, vec2 v2, float a, float b, float c)
{
    return v0 * a + v1 * b + v2 * c;
}
vec3 baryInterpolate(vec3 v0, vec3 v1, vec3 v2, float a, float b, float c)
{
    return v0 * a + v1 * b + v2 * c;
}
vec4 baryInterpolate(vec4 v0, vec4 v1, vec4 v2, float a, float b, float c)
{
    return v0 * a + v1 * b + v2 * c;
}

Vertex interpolate(Vertex v0, Vertex v1, Vertex v2, vec2 baryCoord)
{
    float a = 1 - baryCoord.x - baryCoord.y;
    float b = baryCoord.x;
    float c = baryCoord.y;

    Vertex ret;
    ret.Position =
        baryInterpolate(v0.Position, v1.Position, v2.Position, a, b, c);
    ret.Normal = baryInterpolate(v0.Normal, v1.Normal, v2.Normal, a, b, c);
    ret.Tangent = baryInterpolate(v0.Tangent, v1.Tangent, v2.Tangent, a, b, c);
    ret.TexCoord0 =
        baryInterpolate(v0.TexCoord0, v1.TexCoord0, v2.TexCoord0, a, b, c);

    return ret;
}

vec2 interpolate(vec2 v0, vec2 v1, vec2 v2, vec2 baryCoord)
{
    float a = 1 - baryCoord.x - baryCoord.y;
    float b = baryCoord.x;
    float c = baryCoord.y;

    return baryInterpolate(v0, v1, v2, a, b, c);
}

#endif // SCENE_GEOMETRY_GLSL
