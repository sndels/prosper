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

std::string readFileString(const std::filesystem::path &path);

#endif // PROSPER_CONSTANTS_HPP
