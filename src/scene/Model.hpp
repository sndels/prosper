#ifndef PROSPER_SCENE_MODEL_HPP
#define PROSPER_SCENE_MODEL_HPP

#include "Allocators.hpp"

#include <shader_structs/scene/model_instance_transforms.h>
#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>

namespace scene
{

struct Model
{
    struct SubModel
    {
        uint32_t meshIndex{0xFFFF'FFFF};
        uint32_t materialIndex{0xFFFF'FFFF};
    };

    wheels::Array<SubModel> subModels{gAllocators.world};
};

struct ModelInstance
{
    uint32_t id{0};
    uint32_t modelIndex{0xFFFF'FFFF};
    shader_structs::ModelInstanceTransforms transforms;
    wheels::StrSpan fullName;
};

} // namespace scene

#endif // PROSPER_SCENE_MODEL_HPP
