cmake_minimum_required(VERSION 2.8.7 FATAL_ERROR)
set(CPM_MODULE_NAME "mxnetpredict")
project("${CPM_MODULE_NAME}")

#------------------------------------------------------------------------------
# Required CPM Setup - no need to modify - See: https://github.com/iauns/cpm
#------------------------------------------------------------------------------

set(CPM_DIR "${CMAKE_CURRENT_BINARY_DIR}/cpm_packages" CACHE TYPE STRING)
find_package(Git)
if(NOT GIT_FOUND)
  message(FATAL_ERROR "CPM requires Git.")
endif()
if (NOT EXISTS ${CPM_DIR}/CPM.cmake)
  message(STATUS "Cloning repo (https://github.com/iauns/cpm)")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" clone https://github.com/iauns/cpm ${CPM_DIR}
    RESULT_VARIABLE error_code
    OUTPUT_QUIET ERROR_QUIET)
  if(error_code)
    message(FATAL_ERROR "CPM failed to get the hash for HEAD")
  endif()
endif()
include(${CPM_DIR}/CPM.cmake)

#------------------------------------------------------------------------------
# CPM Modules
#------------------------------------------------------------------------------

CPM_AddModule("cpm_catch"
    GIT_REPOSITORY "https://github.com/bcachet/cpm-catch")

if(NOT DEFINED CMAKECONFIG_PATH)
    CPM_AddModule("cmakeconfig"
        GIT_REPOSITORY "https://github.com/BioroboticsLab/cmakeconfig.git"
        GIT_TAG "master")
else()
    CPM_AddModule("cmakeconfig"
        SOURCE_DIR "${CMAKECONFIG_PATH}")
endif()

if(NOT DEFINED MXNET_PATH)
    CPM_AddModule("cpm_mxnet"
        GIT_REPOSITORY "https://github.com/nebw/cpm_mxnet.git"
        GIT_TAG "master")
else()
    CPM_AddModule("cpm_mxnet"
        SOURCE_DIR "${MXNET_PATH}")
endif()

CPM_InitModule("${CPM_MODULE_NAME}")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

biorobotics_config()

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/mxnetpredictor/
    SYSTEM ${CATCH_INCLUDE_DIR}
    SYSTEM ${OpenCV_INCLUDE_DIRS}
    SYSTEM ${CUDA_INCLUDE_DIRS}
)

add_library(mxnetpredictor STATIC
	"source/MXNetPredictor.cpp"
	"include/mxnetpredictor/MXNetPredictor.h"
)


target_link_libraries(mxnetpredictor
	${CPM_LIBRARIES}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)

CPM_ExportAdditionalIncludeDir("${PROJECT_SOURCE_DIR}/include")

set(test-libs mxnetpredictor)

enable_testing(true)
add_subdirectory(test)
