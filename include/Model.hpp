#ifndef PROSPER_MODEL_HPP
#define PROSPER_MODEL_HPP

// CMake doesn't seem to support MSVC /external -stuff yet
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif // _MSC_VER

#include <glm/glm.hpp>

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

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
