add_subdirectory(dof)
add_subdirectory(rtdi)

set(PROSPER_RENDER_INCLUDES
    ${PROSPER_DOF_INCLUDES}
    ${PROSPER_RTDI_INCLUDES}
    ${CMAKE_CURRENT_LIST_DIR}/ComputePass.hpp
    ${CMAKE_CURRENT_LIST_DIR}/DebugRenderer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/DeferredShading.hpp
    ${CMAKE_CURRENT_LIST_DIR}/ForwardRenderer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/GBufferRenderer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/ImageBasedLighting.hpp
    ${CMAKE_CURRENT_LIST_DIR}/ImGuiRenderer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/LightClustering.hpp
    ${CMAKE_CURRENT_LIST_DIR}/MeshletCuller.hpp
    ${CMAKE_CURRENT_LIST_DIR}/RenderImageCollection.hpp
    ${CMAKE_CURRENT_LIST_DIR}/RenderResourceCollection.hpp
    ${CMAKE_CURRENT_LIST_DIR}/RenderResourceHandle.hpp
    ${CMAKE_CURRENT_LIST_DIR}/RenderResources.hpp
    ${CMAKE_CURRENT_LIST_DIR}/RenderTargets.hpp
    ${CMAKE_CURRENT_LIST_DIR}/RtReference.hpp
    ${CMAKE_CURRENT_LIST_DIR}/SkyboxRenderer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/TemporalAntiAliasing.hpp
    ${CMAKE_CURRENT_LIST_DIR}/TextureDebug.hpp
    ${CMAKE_CURRENT_LIST_DIR}/ToneMap.hpp
    PARENT_SCOPE
)

set(PROSPER_RENDER_SOURCES
    ${PROSPER_DOF_SOURCES}
    ${PROSPER_RTDI_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/ComputePass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/DebugRenderer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/DeferredShading.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ForwardRenderer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/GBufferRenderer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ImageBasedLighting.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ImGuiRenderer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/LightClustering.cpp
    ${CMAKE_CURRENT_LIST_DIR}/MeshletCuller.cpp
    ${CMAKE_CURRENT_LIST_DIR}/RenderImageCollection.cpp
    ${CMAKE_CURRENT_LIST_DIR}/RenderResources.cpp
    ${CMAKE_CURRENT_LIST_DIR}/RenderTargets.cpp
    ${CMAKE_CURRENT_LIST_DIR}/RtReference.cpp
    ${CMAKE_CURRENT_LIST_DIR}/SkyboxRenderer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/TemporalAntiAliasing.cpp
    ${CMAKE_CURRENT_LIST_DIR}/TextureDebug.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ToneMap.cpp
    PARENT_SCOPE
)
