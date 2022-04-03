#ifndef PROSPER_MODEL_HPP
#define PROSPER_MODEL_HPP

#include <glm/glm.hpp>

struct Model
{
    struct SubModel
    {
        uint32_t meshID{0xFFFFFFFF};
        uint32_t materialID{0xFFFFFFFF};
    };

    std::vector<SubModel> subModels;
};

struct ModelInstance
{
    struct PCBlock
    {
        uint32_t modelInstanceID{0xFFFFFFFF};
        uint32_t materialID{0xFFFFFFFF};
    };

    struct Transforms
    {
        glm::mat4 modelToWorld{1.f};
        glm::mat4 normalToWorld{1.f};
    };

    uint32_t id{0};
    uint32_t modelID{0xFFFFFFFF};
    Transforms transforms;
};

#endif // PROSPER_MODEL_HPP
