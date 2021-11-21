#ifndef PROSPER_CONSTANTS_HPP
#define PROSPER_CONSTANTS_HPP

#include "vulkan.hpp"

#include <cstddef>
#include <string>
#include <vector>

const size_t MAX_FRAMES_IN_FLIGHT = 2;

std::string resPath(const std::string &res);
std::string binPath(const std::string &bin);

std::vector<std::byte> readFileBytes(const std::string &filename);

vk::ShaderModule createShaderModule(
    const vk::Device device, const std::vector<std::byte> &spv);

#endif // PROSPER_CONSTANTS_HPP