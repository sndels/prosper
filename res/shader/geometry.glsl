// Unbounded array requires GL_EXT_nonuniform_qualifier even if it is indexed
// with a uniform index
layout(std430, set = 4, binding = 0) readonly buffer VertexBuffers
{
    float data[];
}
vertexBuffers[];

// Unbounded array requires GL_EXT_nonuniform_qualifier even if it is indexed
// with a uniform index
layout(std430, set = 5, binding = 0) readonly buffer IndexBuffers
{
    uint data[];
}
indexBuffers[];

struct Vertex
{
    vec3 Position;
    vec3 Normal;
    vec4 Tangent;
    vec2 TexCoord0;
};
#define VERTEX_FLOATS (3 + 3 + 4 + 2)
#define VERTEX_POS_FLOAT_OFFSET 0
#define VERTEX_NORMAL_FLOAT_OFFSET 3
#define VERTEX_TANGENT_FLOAT_OFFSET (3 + 3)
#define VERTEX_TEXCOORD0_FLOAT_OFFSET (3 + 3 + 4)

Vertex loadVertex(uint meshID, uint index)
{
    uint vertexIndex = indexBuffers[meshID].data[index];
    uint vertexOffset = vertexIndex * VERTEX_FLOATS;

    Vertex ret;

    ret.Position = vec3(
        vertexBuffers[meshID].data[vertexOffset + VERTEX_POS_FLOAT_OFFSET + 0],
        vertexBuffers[meshID].data[vertexOffset + VERTEX_POS_FLOAT_OFFSET + 1],
        vertexBuffers[meshID].data[vertexOffset + VERTEX_POS_FLOAT_OFFSET + 2]);
    ret.Normal = vec3(
        vertexBuffers[meshID]
            .data[vertexOffset + VERTEX_NORMAL_FLOAT_OFFSET + 0],
        vertexBuffers[meshID]
            .data[vertexOffset + VERTEX_NORMAL_FLOAT_OFFSET + 1],
        vertexBuffers[meshID]
            .data[vertexOffset + VERTEX_NORMAL_FLOAT_OFFSET + 2]);
    ret.Tangent = vec4(
        vertexBuffers[meshID]
            .data[vertexOffset + VERTEX_TANGENT_FLOAT_OFFSET + 0],
        vertexBuffers[meshID]
            .data[vertexOffset + VERTEX_TANGENT_FLOAT_OFFSET + 1],
        vertexBuffers[meshID]
            .data[vertexOffset + VERTEX_TANGENT_FLOAT_OFFSET + 2],
        vertexBuffers[meshID]
            .data[vertexOffset + VERTEX_TANGENT_FLOAT_OFFSET + 3]);
    ret.TexCoord0 = vec2(
        vertexBuffers[meshID]
            .data[vertexOffset + VERTEX_TEXCOORD0_FLOAT_OFFSET + 0],
        vertexBuffers[meshID]
            .data[vertexOffset + VERTEX_TEXCOORD0_FLOAT_OFFSET + 1]);

    return ret;
}
