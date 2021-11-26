#ifndef PROSPER_CONSTANTS_HPP
#define PROSPER_CONSTANTS_HPP

#include "vulkan.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

const size_t MAX_FRAMES_IN_FLIGHT = 2;

std::filesystem::path resPath(const std::filesystem::path &path);
std::filesystem::path binPath(const std::filesystem::path &path);

std::vector<std::byte> readFileBytes(const std::filesystem::path &path);

vk::ShaderModule createShaderModule(
    const vk::Device device, const std::vector<std::byte> &spv);

#endif // PROSPER_CONSTANTS_HPP
