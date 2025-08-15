#include "DrawType.hpp"

namespace scene
{

const wheels::StaticArray<const char *, static_cast<size_t>(DrawType::Count)>
    sDrawTypeNames{{DRAW_TYPES_STRS}};

} // namespace scene
