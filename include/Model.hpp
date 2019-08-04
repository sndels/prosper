#ifndef PROSPER_MODEL_HPP
#define PROSPER_MODEL_HPP

#include "Mesh.hpp"

struct Model
{
    std::shared_ptr<Device> _device = nullptr;
    std::vector<Mesh> _meshes;
    // TODO: Materials for meshes here instead of Mesh?
};

#endif // PROSPER_MODEL_HPP
