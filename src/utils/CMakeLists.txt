set(PROSPER_UTILS_INCLUDES
    ${CMAKE_CURRENT_LIST_DIR}/Dds.hpp
    ${CMAKE_CURRENT_LIST_DIR}/ForEach.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Fwd.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Hashes.hpp
    ${CMAKE_CURRENT_LIST_DIR}/InputHandler.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Ktx.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Logger.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Profiler.hpp
    ${CMAKE_CURRENT_LIST_DIR}/SceneStats.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Timer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Ui.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Utils.hpp
    PARENT_SCOPE
)

set(PROSPER_UTILS_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/Dds.cpp
    ${CMAKE_CURRENT_LIST_DIR}/InputHandler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Ktx.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Logger.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Profiler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Timer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
    PARENT_SCOPE
)
