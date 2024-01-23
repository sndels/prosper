#ifndef PROSPER_SCENE_MODEL_HPP
#define PROSPER_SCENE_MODEL_HPP

#include <glm/glm.hpp>

#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>

struct Model
{
    struct SubModel
    {
        uint32_t meshID{0xFFFFFFFF};
        uint32_t materialID{0xFFFFFFFF};
    };

    wheels::Array<SubModel> subModels;

    Model(wheels::Allocator &alloc)
    : subModels{alloc}
    {
    }
};

struct ModelInstance
{
    struct Transforms
    {
        glm::mat3x4 modelToWorld{1.f};
        glm::mat3x4 normalToWorld{1.f};
    };

    uint32_t id{0};
    uint32_t modelID{0xFFFFFFFF};
    Transforms transforms;
};

#endif // PROSPER_SCENE_MODEL_HPP
