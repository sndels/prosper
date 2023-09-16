#ifndef PROSPER_SCENE_DEBUG_DRAW_TYPES_HPP
#define PROSPER_SCENE_DEBUG_DRAW_TYPES_HPP

#include "../utils/ForEach.hpp"

#define DEBUG_DRAW_TYPES                                                       \
    PrimitiveID, MeshID, MaterialID, Position, ShadingNormal, TexCoord0,       \
        Albedo, Roughness, Metallic

#define DEBUG_DRAW_TYPES_AND_COUNT DEBUG_DRAW_TYPES, Count

#define DEBUG_DRAW_TYPES_STRINGIFY(t) #t,

#define DEBUG_DRAW_TYPES_STRS                                                  \
    FOR_EACH(DEBUG_DRAW_TYPES_STRINGIFY, DEBUG_DRAW_TYPES)

#endif // PROSPER_SCENE_DEBUG_DRAW_TYPES_HPP
