#ifndef PROSPER_CONSTANTS_HPP
#define PROSPER_CONSTANTS_HPP

#include <cstddef>
#include <string>

const size_t MAX_FRAMES_IN_FLIGHT = 2;

std::string resPath(const std::string& res);
std::string binPath(const std::string& bin);

#endif // PROSPER_CONSTANTS_HPP
