add_subdirectory(gfx)
add_subdirectory(render)
add_subdirectory(scene)
add_subdirectory(utils)

set(PROSPER_INCLUDE_DIR
    ${CMAKE_CURRENT_LIST_DIR}
    PARENT_SCOPE
)

set(PROSPER_INCLUDES
    ${PROSPER_GFX_INCLUDES}
    ${PROSPER_RENDER_INCLUDES}
    ${PROSPER_SCENE_INCLUDES}
    ${PROSPER_UTIL_INCLUDES}
    ${CMAKE_CURRENT_LIST_DIR}/Allocators.hpp
    ${CMAKE_CURRENT_LIST_DIR}/App.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Window.hpp
    PARENT_SCOPE
)

set(PROSPER_SOURCES
    ${PROSPER_GFX_SOURCES}
    ${PROSPER_RENDER_SOURCES}
    ${PROSPER_SCENE_SOURCES}
    ${PROSPER_UTILS_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/Allocators.cpp
    ${CMAKE_CURRENT_LIST_DIR}/App.cpp
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Window.cpp
    PARENT_SCOPE
)
