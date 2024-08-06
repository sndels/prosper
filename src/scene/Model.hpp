#ifndef PROSPER_SCENE_MODEL_HPP
#define PROSPER_SCENE_MODEL_HPP

#include "../Allocators.hpp"

#include <glm/glm.hpp>
#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>

struct Model
{
    struct SubModel
    {
        uint32_t meshID{0xFFFF'FFFF};
        uint32_t materialID{0xFFFF'FFFF};
    };

    wheels::Array<SubModel> subModels{gAllocators.world};
};

struct ModelInstance
{
    struct Transforms
    {
        glm::mat3x4 modelToWorld{1.f};
        glm::mat3x4 normalToWorld{1.f};
    };

    uint32_t id{0};
    uint32_t modelID{0xFFFF'FFFF};
    Transforms transforms;
    wheels::StrSpan fullName;
};

#endif // PROSPER_SCENE_MODEL_HPP
