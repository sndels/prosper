#ifndef WHEELS_UTILS_UI_HPP
#define WHEELS_UTILS_UI_HPP

#include <imgui.h>
#include <type_traits>
#include <wheels/containers/static_array.hpp>

// Creates a dropdown for the value and returns true if it was changed
template <typename Enum, size_t N>
bool enumDropdown(
    const char *label, Enum &value,
    const wheels::StaticArray<const char *, N> &variantNames)
{
    using EnumType = std::underlying_type_t<Enum>;

    bool changed = false;
    auto *currentType = reinterpret_cast<EnumType *>(&value);
    if (ImGui::BeginCombo(label, variantNames[*currentType]))
    {
        for (EnumType i = 0; i < static_cast<EnumType>(Enum::Count); ++i)
        {
            bool selected = *currentType == i;
            if (ImGui::Selectable(variantNames[i], &selected))
            {
                value = static_cast<Enum>(i);
                changed = true;
            }
        }
        ImGui::EndCombo();
    }

    return changed;
}

// Returns true if the value was changed
inline bool sliderU32(
    const char *label, uint32_t *v, uint32_t v_min, uint32_t v_max)
{
    return ImGui::SliderScalar(label, ImGuiDataType_U32, v, &v_min, &v_max);
}

#endif // WHEELS_UTILS_UI_HPP
