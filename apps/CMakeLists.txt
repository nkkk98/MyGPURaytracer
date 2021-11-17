# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 11)
# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
include_directories(.)
########################################
# CUDA Setup
########################################
find_package(CUDA 10 REQUIRED)
include(${CMAKE_MODULE_PATH}/CUDAComputesList.cmake)

list(APPEND CUDA_NVCC_FLAGS ${CUDA_GENERATE_CODE})
list(APPEND CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CUDA_VERBOSE_BUILD ON)

if(WIN32)
    # Set up include and lib paths
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "Host side compiler used by NVCC" FORCE)
endif(WIN32)
########################################

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(OpenGL REQUIRED)

if(UNIX)
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_gl_LIBRARY})
else(UNIX)
    set(EXTERNAL "external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

add_subdirectory(stream_compaction)

include_directories(.)

set(headers
    src/main.h
    src/image.h
    src/interactions.h
    src/intersections.h
    src/glslUtility.hpp
    src/pathtrace.h
    src/scene.h
    src/sceneStructs.h
    src/preview.h
    src/utilities.h
    src/tiny_obj_loader.h
    src/timer.h
    )

set(sources
    src/main.cpp
    src/stb.cpp
    src/image.cpp
    src/glslUtility.cpp
    src/pathtrace.cu
    src/scene.cpp
    src/preview.cpp
    src/utilities.cpp
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})


#cuda_add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})
add_subdirectory(utils)

macro(oidn_add_app APP_NAME)
  add_executable(${APP_NAME} ${ARGN} ${OIDN_RESOURCE})
  target_link_libraries(${APP_NAME} PRIVATE common utils ${PROJECT_NAME})
  install(TARGETS ${APP_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT apps)
endmacro()


oidn_add_app(oidnDenoise oidnDenoise.cpp)
oidn_add_app(oidnBenchmark oidnBenchmark.cpp)
oidn_add_app(oidnTest oidnTest.cpp catch.hpp)

set(APP_NAME oidnRayTracer)
cuda_add_executable(${APP_NAME} ${sources} ${headers})
target_link_libraries(${APP_NAME} common utils ${LIBRARIES} ${PROJECT_NAME})
install(TARGETS ${APP_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT apps)

add_custom_command(
    TARGET ${APP_NAME}
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/textures
        ${CMAKE_BINARY_DIR}/textures
    )
add_custom_command(
    TARGET ${APP_NAME}
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/scenes
        ${CMAKE_BINARY_DIR}/scenes
    )
add_custom_command(
    TARGET ${APP_NAME}
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/models
        ${CMAKE_BINARY_DIR}/models
    )