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
    // All of these offsets are into the data interpreted as a u32 'array'
    // offset of 0xFFFFFFFF signals an unused attribute
    uint indicesOffset;
    uint positionsOffset;
    uint normalsOffset;
    uint tangentsOffset;
    uint texCoord0sOffset;
    uint meshletsOffset;
    uint meshletVerticesOffset;
    uint meshletTrianglesByteOffset;
    uint usesShortIndices;
};
layout(std430, set = GEOMETRY_SET, binding = 0) readonly buffer
    GeometryMetadatas
{
    GeometryMetadata data[];
}
geometryMetadatas;

// Aliased binds of the same SSBOs
layout(std430, set = GEOMETRY_SET, binding = 1) readonly buffer
    GeometryBuffersU32
{
    uint data[];
}
geometryBuffersU32[];
layout(std430, set = GEOMETRY_SET, binding = 1) readonly buffer
    GeometryBuffersU16
{
    uint16_t data[];
}
geometryBuffersU16[];
layout(std430, set = GEOMETRY_SET, binding = 1) readonly buffer
    GeometryBuffersU8
{
    uint8_t data[];
}
geometryBuffersU8[];
layout(std430, set = GEOMETRY_SET, binding = 1) readonly buffer
    GeometryBuffersF32
{
    float data[];
}
geometryBuffersF32[];

#ifdef NON_UNIFORM_GEOMETRY_BUFFER_INDICES
#define GET_GEOMETRY_BUFFER_U32(index) geometryBuffersU32[nonuniformEXT(index)]
#define GET_GEOMETRY_BUFFER_U16(index) geometryBuffersU16[nonuniformEXT(index)]
#define GET_GEOMETRY_BUFFER_U8(index) geometryBuffersU8[nonuniformEXT(index)]
#define GET_GEOMETRY_BUFFER_F32(index) geometryBuffersF32[nonuniformEXT(index)]
#else // !NON_UNIFORM_GEOMETRY_BUFFER_INDICES
#define GET_GEOMETRY_BUFFER_U32(index) geometryBuffersU32[index]
#define GET_GEOMETRY_BUFFER_U16(index) geometryBuffersU16[index]
#define GET_GEOMETRY_BUFFER_U8(index) geometryBuffersU8[index]
#define GET_GEOMETRY_BUFFER_F32(index) geometryBuffersF32[index]
#endif // NON_UNIFORM_GEOMETRY_BUFFER_INDICES

uint loadIndex(
    uint bufferIndex, uint bufferOffset, uint index, uint usesShortIndices)
{
    if (usesShortIndices == 1)
    {
        return uint(
            GET_GEOMETRY_BUFFER_U16(bufferIndex).data[bufferOffset + index]);
    }
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

vec2 loadVec2(uint bufferIndex, uint bufferOffset, uint index)
{
    return bufferOffset < 0xFFFFFFFF
               ? vec2(
                     loadFloat(bufferIndex, bufferOffset, 2 * index),
                     loadFloat(bufferIndex, bufferOffset, 2 * index + 1))
               : vec2(0);
}

vec3 loadVec3(uint bufferIndex, uint bufferOffset, uint index)
{
    return bufferOffset < 0xFFFFFFFF
               ? vec3(
                     loadFloat(bufferIndex, bufferOffset, 3 * index),
                     loadFloat(bufferIndex, bufferOffset, 3 * index + 1),
                     loadFloat(bufferIndex, bufferOffset, 3 * index + 2))
               : vec3(0);
}

vec4 loadVec4(uint bufferIndex, uint bufferOffset, uint index)
{
    return bufferOffset < 0xFFFFFFFF
               ? vec4(
                     loadFloat(bufferIndex, bufferOffset, 4 * index),
                     loadFloat(bufferIndex, bufferOffset, 4 * index + 1),
                     loadFloat(bufferIndex, bufferOffset, 4 * index + 2),
                     loadFloat(bufferIndex, bufferOffset, 4 * index + 3))
               : vec4(0);
}

struct MeshletInfo
{
    uint vertexOffset;
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
        loadVec3(metadata.bufferIndex, metadata.positionsOffset, index);
    ret.Normal = loadVec3(metadata.bufferIndex, metadata.normalsOffset, index);
    ret.Tangent =
        loadVec4(metadata.bufferIndex, metadata.tangentsOffset, index);
    ret.TexCoord0 =
        loadVec2(metadata.bufferIndex, metadata.texCoord0sOffset, index);

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

    return loadVec2(
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
