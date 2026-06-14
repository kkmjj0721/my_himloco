cmake_minimum_required(VERSION 3.10)
project(sim2real_deploy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-O3)

get_filename_component(SIM2REAL_DEPLOY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

set(SIM2REAL_POLICY_DIR "${SIM2REAL_DEPLOY_DIR}/policy" CACHE PATH "Deploy policy directory")
set(SIM2REAL_CONFIG_DIR "${SIM2REAL_DEPLOY_DIR}/config" CACHE PATH "Deploy robot config directory")
# The Unitree actuator SDK is vendored directly inside the deploy library tree so the
# project is self-contained and builds without any external checkout.
get_filename_component(UNITREE_ACTUATOR_SDK_DEFAULT_VENDOR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/unitree_actuator_sdk" ABSOLUTE)
get_filename_component(UNITREE_ACTUATOR_SDK_DEFAULT_EXTERNAL_ROOT "${SIM2REAL_DEPLOY_DIR}/../unitree_actuator_sdk" ABSOLUTE)
set(UNITREE_ACTUATOR_SDK_VENDOR_ROOT "${UNITREE_ACTUATOR_SDK_DEFAULT_VENDOR_ROOT}" CACHE PATH "Vendored Unitree actuator SDK root")
set(UNITREE_ACTUATOR_SDK_EXTERNAL_ROOT "${UNITREE_ACTUATOR_SDK_DEFAULT_EXTERNAL_ROOT}" CACHE PATH "External Unitree actuator SDK fallback root")
set(UNITREE_ACTUATOR_SDK_ROOT "" CACHE PATH "Optional Unitree actuator SDK root override; empty auto-selects vendored SDK, then external fallback")
set(SIM2REAL_ENABLE_TORCH "AUTO" CACHE STRING "Torch inference backend: AUTO, ON, or OFF")
set(SIM2REAL_ENABLE_ONNX "AUTO" CACHE STRING "ONNX Runtime inference backend: AUTO, ON, or OFF")
set_property(CACHE SIM2REAL_ENABLE_TORCH PROPERTY STRINGS AUTO ON OFF)
set_property(CACHE SIM2REAL_ENABLE_ONNX PROPERTY STRINGS AUTO ON OFF)

find_package(Threads REQUIRED)
find_package(Eigen3 QUIET)
find_package(OpenCV QUIET)
find_package(PCL QUIET COMPONENTS common io)
find_package(yaml-cpp QUIET)
find_package(TBB QUIET)
find_package(Torch QUIET)

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ENV ONNXRUNTIME_ROOT
    PATH_SUFFIXES include
)
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS ENV ONNXRUNTIME_ROOT
    PATH_SUFFIXES lib lib64
)

if(UNITREE_ACTUATOR_SDK_ROOT
        AND NOT UNITREE_ACTUATOR_SDK_ROOT STREQUAL UNITREE_ACTUATOR_SDK_VENDOR_ROOT
        AND NOT UNITREE_ACTUATOR_SDK_ROOT STREQUAL UNITREE_ACTUATOR_SDK_EXTERNAL_ROOT)
    set(UNITREE_ACTUATOR_SDK_SELECTED_ROOT "${UNITREE_ACTUATOR_SDK_ROOT}")
    set(UNITREE_ACTUATOR_SDK_SOURCE "custom override")
elseif(EXISTS "${UNITREE_ACTUATOR_SDK_VENDOR_ROOT}")
    set(UNITREE_ACTUATOR_SDK_SELECTED_ROOT "${UNITREE_ACTUATOR_SDK_VENDOR_ROOT}")
    set(UNITREE_ACTUATOR_SDK_SOURCE "vendored")
else()
    set(UNITREE_ACTUATOR_SDK_SELECTED_ROOT "${UNITREE_ACTUATOR_SDK_EXTERNAL_ROOT}")
    set(UNITREE_ACTUATOR_SDK_SOURCE "external fallback")
endif()
set(UNITREE_ACTUATOR_SDK_ROOT "${UNITREE_ACTUATOR_SDK_SELECTED_ROOT}" CACHE PATH "Selected Unitree actuator SDK root" FORCE)

set(UNITREE_ACTUATOR_SDK_INCLUDE_DIR "${UNITREE_ACTUATOR_SDK_SELECTED_ROOT}/include")
set(UNITREE_ACTUATOR_SDK_LIB_DIR "${UNITREE_ACTUATOR_SDK_SELECTED_ROOT}/lib")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(UNITREE_ACTUATOR_SDK_LIBRARY "${UNITREE_ACTUATOR_SDK_LIB_DIR}/libUnitreeMotorSDK_Arm64.so")
    set(UNITREE_ACTUATOR_SDK_ARCH "arm64")
else()
    set(UNITREE_ACTUATOR_SDK_LIBRARY "${UNITREE_ACTUATOR_SDK_LIB_DIR}/libUnitreeMotorSDK_Linux64.so")
    set(UNITREE_ACTUATOR_SDK_ARCH "x86_64")
endif()

if(NOT EXISTS "${UNITREE_ACTUATOR_SDK_INCLUDE_DIR}/unitreeMotor/unitreeMotor.h")
    message(FATAL_ERROR "Unitree actuator SDK headers not found under ${UNITREE_ACTUATOR_SDK_INCLUDE_DIR}")
endif()

if(NOT EXISTS "${UNITREE_ACTUATOR_SDK_LIBRARY}")
    message(FATAL_ERROR "Unitree actuator SDK ${UNITREE_ACTUATOR_SDK_ARCH} library not found: ${UNITREE_ACTUATOR_SDK_LIBRARY}")
endif()

add_library(UnitreeMotorSDK SHARED IMPORTED GLOBAL)
set_target_properties(UnitreeMotorSDK PROPERTIES
    IMPORTED_LOCATION "${UNITREE_ACTUATOR_SDK_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${UNITREE_ACTUATOR_SDK_INCLUDE_DIR}"
)

add_library(sim2real_deploy_common INTERFACE)
target_include_directories(sim2real_deploy_common INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/fsm_robot"
    "${CMAKE_CURRENT_SOURCE_DIR}/library/core/inference_runtime"
    "${CMAKE_CURRENT_SOURCE_DIR}/library/core/logger"
    "${CMAKE_CURRENT_SOURCE_DIR}/library/core/loop"
    "${CMAKE_CURRENT_SOURCE_DIR}/library/core/observation_buffer"
    "${CMAKE_CURRENT_SOURCE_DIR}/library/core/rl_sdk"
    "${CMAKE_CURRENT_SOURCE_DIR}/library/core/vector_math"
    "${UNITREE_ACTUATOR_SDK_INCLUDE_DIR}"
)
target_compile_definitions(sim2real_deploy_common INTERFACE
    SIM2REAL_POLICY_DIR="${SIM2REAL_POLICY_DIR}"
    SIM2REAL_CONFIG_DIR="${SIM2REAL_CONFIG_DIR}"
)

add_library(sim2real_deploy_inference INTERFACE)

set(SIM2REAL_TORCH_ENABLED FALSE)
if(SIM2REAL_ENABLE_TORCH STREQUAL "ON" OR
        (SIM2REAL_ENABLE_TORCH STREQUAL "AUTO" AND Torch_FOUND))
    if(Torch_FOUND)
        set(SIM2REAL_TORCH_ENABLED TRUE)
        target_compile_definitions(sim2real_deploy_inference INTERFACE USE_TORCH)
        target_link_libraries(sim2real_deploy_inference INTERFACE "${TORCH_LIBRARIES}")
        if(TORCH_CXX_FLAGS)
            separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
            target_compile_options(sim2real_deploy_inference INTERFACE ${TORCH_CXX_FLAGS_LIST})
        endif()
        message(STATUS "Torch inference backend enabled")
    endif()
endif()
if(SIM2REAL_ENABLE_TORCH STREQUAL "AUTO" AND NOT Torch_FOUND)
    message(STATUS "Torch inference backend disabled: LibTorch not found")
endif()

set(SIM2REAL_ONNX_FOUND FALSE)
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    set(SIM2REAL_ONNX_FOUND TRUE)
endif()

set(SIM2REAL_ONNX_ENABLED FALSE)
if(SIM2REAL_ENABLE_ONNX STREQUAL "ON" OR
        (SIM2REAL_ENABLE_ONNX STREQUAL "AUTO" AND SIM2REAL_ONNX_FOUND))
    if(SIM2REAL_ONNX_FOUND)
        set(SIM2REAL_ONNX_ENABLED TRUE)
        target_compile_definitions(sim2real_deploy_inference INTERFACE USE_ONNX)
        target_include_directories(sim2real_deploy_inference INTERFACE "${ONNXRUNTIME_INCLUDE_DIR}")
        target_link_libraries(sim2real_deploy_inference INTERFACE "${ONNXRUNTIME_LIBRARY}")
        message(STATUS "ONNX Runtime inference backend enabled")
    endif()
endif()
if(SIM2REAL_ENABLE_ONNX STREQUAL "AUTO" AND NOT SIM2REAL_ONNX_FOUND)
    message(STATUS "ONNX Runtime inference backend disabled: headers/library not found")
endif()

if(Eigen3_FOUND)
    target_link_libraries(sim2real_deploy_common INTERFACE Eigen3::Eigen)
    message(STATUS "Eigen3 found")
else()
    message(STATUS "Eigen3 not found; continuing because current deploy targets do not require it")
endif()

if(OpenCV_FOUND)
    target_include_directories(sim2real_deploy_common INTERFACE ${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(STATUS "OpenCV not found; continuing because current deploy targets do not require it")
endif()

if(PCL_FOUND)
    target_include_directories(sim2real_deploy_common INTERFACE ${PCL_INCLUDE_DIRS})
    message(STATUS "PCL found: ${PCL_VERSION}")
else()
    message(STATUS "PCL not found; continuing because current deploy targets do not require it")
endif()

set(SIM2REAL_DEPLOY_EXPECTED_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/rl_real_my_robot.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/unitree_motor_adapter.cpp"
)

set(SIM2REAL_DEPLOY_CORE_SOURCES)
foreach(core_source IN ITEMS
        "${CMAKE_CURRENT_SOURCE_DIR}/library/core/observation_buffer/observation_buffer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/library/core/rl_sdk/rl_sdk.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/library/core/inference_runtime/inference_runtime.cpp")
    if(EXISTS "${core_source}")
        list(APPEND SIM2REAL_DEPLOY_CORE_SOURCES "${core_source}")
    endif()
endforeach()

set(SIM2REAL_DEPLOY_MISSING_SOURCES)
foreach(expected_source IN LISTS SIM2REAL_DEPLOY_EXPECTED_SOURCES)
    if(NOT EXISTS "${expected_source}")
        list(APPEND SIM2REAL_DEPLOY_MISSING_SOURCES "${expected_source}")
    endif()
endforeach()

if(SIM2REAL_DEPLOY_MISSING_SOURCES)
    message(WARNING
        "Deploy main target disabled because expected implementation sources are missing: "
        "${SIM2REAL_DEPLOY_MISSING_SOURCES}"
    )
else()
    set(SIM2REAL_DEPLOY_MISSING_DEPS)
    if(NOT yaml-cpp_FOUND AND NOT TARGET yaml-cpp)
        list(APPEND SIM2REAL_DEPLOY_MISSING_DEPS "yaml-cpp")
    endif()
    if(NOT TBB_FOUND AND NOT TARGET TBB::tbb)
        list(APPEND SIM2REAL_DEPLOY_MISSING_DEPS "TBB")
    endif()
    if(SIM2REAL_ENABLE_TORCH STREQUAL "ON" AND NOT SIM2REAL_TORCH_ENABLED)
        list(APPEND SIM2REAL_DEPLOY_MISSING_DEPS "LibTorch (SIM2REAL_ENABLE_TORCH=ON)")
    endif()
    if(SIM2REAL_ENABLE_ONNX STREQUAL "ON" AND NOT SIM2REAL_ONNX_ENABLED)
        list(APPEND SIM2REAL_DEPLOY_MISSING_DEPS "ONNX Runtime (SIM2REAL_ENABLE_ONNX=ON)")
    endif()

    if(SIM2REAL_DEPLOY_MISSING_DEPS)
        message(WARNING "Deploy main target disabled because required dependencies are missing: ${SIM2REAL_DEPLOY_MISSING_DEPS}")
    else()
        add_executable(main
            ${SIM2REAL_DEPLOY_EXPECTED_SOURCES}
            ${SIM2REAL_DEPLOY_CORE_SOURCES}
        )
        target_compile_features(main PRIVATE cxx_std_17)
        target_link_libraries(main PRIVATE
            sim2real_deploy_common
            sim2real_deploy_inference
            UnitreeMotorSDK
            Threads::Threads
        )

        if(TARGET yaml-cpp)
            target_link_libraries(main PRIVATE yaml-cpp)
        else()
            target_link_libraries(main PRIVATE ${YAML_CPP_LIBRARIES})
        endif()

        if(TARGET TBB::tbb)
            target_link_libraries(main PRIVATE TBB::tbb)
        else()
            target_link_libraries(main PRIVATE ${TBB_LIBRARIES})
        endif()

        if(OpenCV_FOUND)
            target_link_libraries(main PRIVATE ${OpenCV_LIBS})
        endif()

        if(PCL_FOUND)
            target_link_libraries(main PRIVATE ${PCL_LIBRARIES})
        endif()

        set_target_properties(main PROPERTIES
            BUILD_RPATH "${UNITREE_ACTUATOR_SDK_LIB_DIR}"
            INSTALL_RPATH "${UNITREE_ACTUATOR_SDK_LIB_DIR}"
        )
    endif()
endif()

message(STATUS "Unitree actuator SDK source: ${UNITREE_ACTUATOR_SDK_SOURCE}")
message(STATUS "Unitree actuator SDK root: ${UNITREE_ACTUATOR_SDK_SELECTED_ROOT}")
message(STATUS "Unitree actuator SDK arch: ${UNITREE_ACTUATOR_SDK_ARCH}")
message(STATUS "Unitree actuator SDK library: ${UNITREE_ACTUATOR_SDK_LIBRARY}")
message(STATUS "Deploy policy dir: ${SIM2REAL_POLICY_DIR}")
message(STATUS "Deploy config dir: ${SIM2REAL_CONFIG_DIR}")
