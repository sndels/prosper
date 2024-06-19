#ifndef PROSPER_SCENE_DRAW_TYPE_HPP
#define PROSPER_SCENE_DRAW_TYPE_HPP

#include "../utils/ForEach.hpp"
#include <wheels/containers/static_array.hpp>

#define DRAW_TYPES                                                             \
    Default, PrimitiveID, MeshletID, MeshID, MaterialID, Position,             \
        ShadingNormal, GeometryNormal, TexCoord0, Albedo, Roughness, Metallic

#define DRAW_TYPES_AND_COUNT DRAW_TYPES, Count

#define DRAW_TYPES_STRINGIFY(t) #t,

#define DRAW_TYPES_STRS FOR_EACH(DRAW_TYPES_STRINGIFY, DRAW_TYPES)

enum class DrawType : uint32_t
{
    DRAW_TYPES_AND_COUNT
};

extern const wheels::StaticArray<
    const char *, static_cast<size_t>(DrawType::Count)>
    sDrawTypeNames;

#endif // PROSPER_SCENE_DRAW_TYPE_HPP
