set(PROSPER_SCENE_INCLUDES
    ${CMAKE_CURRENT_LIST_DIR}/Animations.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Accessors.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Camera.hpp
    ${CMAKE_CURRENT_LIST_DIR}/DebugGeometry.hpp
    ${CMAKE_CURRENT_LIST_DIR}/DeferredLoadingContext.hpp
    ${CMAKE_CURRENT_LIST_DIR}/DrawType.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Fwd.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Light.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Material.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Mesh.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Model.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Scene.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Texture.hpp
    ${CMAKE_CURRENT_LIST_DIR}/World.hpp
    ${CMAKE_CURRENT_LIST_DIR}/WorldData.hpp
    ${CMAKE_CURRENT_LIST_DIR}/WorldRenderStructs.hpp
    PARENT_SCOPE
)

set(PROSPER_SCENE_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/Accessors.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Camera.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cgltfImplementation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/DebugGeometry.cpp
    ${CMAKE_CURRENT_LIST_DIR}/DeferredLoadingContext.cpp
    ${CMAKE_CURRENT_LIST_DIR}/DrawType.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Light.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Texture.cpp
    ${CMAKE_CURRENT_LIST_DIR}/stbImplementation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/World.cpp
    ${CMAKE_CURRENT_LIST_DIR}/WorldData.cpp
    PARENT_SCOPE
)
