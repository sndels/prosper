#ifndef WHEELS_UTILS_UI_HPP
#define WHEELS_UTILS_UI_HPP

#include <array>
#include <imgui.h>

// Creates a dropdown for the value and returns true if it was changed
template <typename Enum, size_t N>
bool enumDropdown(
    const char *label, Enum &value,
    const std::array<const char *, N> &variantNames)
{
    bool changed = false;
    auto *currentType = reinterpret_cast<uint32_t *>(&value);
    if (ImGui::BeginCombo(label, variantNames[*currentType]))
    {
        for (auto i = 0u; i < static_cast<uint32_t>(Enum::Count); ++i)
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
#endif // WHEELS_UTILS_UI_HPP
