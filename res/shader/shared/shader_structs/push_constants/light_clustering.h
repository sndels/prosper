#ifndef SHADER_STRUCTS_PUSH_CONSTANTS_LIGHT_CLUSTERING_H
#define SHADER_STRUCTS_PUSH_CONSTANTS_LIGHT_CLUSTERING_H

#include "../fields.h"

struct LightClusteringPC
{
    STRUCT_FIELD_GLM(uvec2, resolution, {});
};

#endif // SHADER_STRUCTS_PUSH_CONSTANTS_LIGHT_CLUSTERING_H
