#ifndef PROSPER_MODEL_HPP
#define PROSPER_MODEL_HPP

struct Model
{
    struct SubModel
    {
        uint32_t meshID{0};
        uint32_t materialID{0};
    };

    std::vector<SubModel> subModels;
};

#endif // PROSPER_MODEL_HPP
