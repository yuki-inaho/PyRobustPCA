cmake_minimum_required(VERSION 3.18...3.22)

project(PyRobustPCA)

# Set build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release")
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

add_compile_options(-Wall -Wextra)
add_compile_options(
  "$<$<CONFIG:Debug>:-Og>"
  "$<$<CONFIG:Release>:-O3>"
  "$<$<CONFIG:None>:-O3>")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Setup for scikit-build
if(NOT SKBUILD)
  message(WARNING "This CMake file should be executed via scikit-build. "
    "Please run\n$ pip install .")
endif()

if(SKBUILD)
  # Constrain FindPython to find the Python version used by scikit-build
  set(Python_VERSION "${PYTHON_VERSION_STRING}")
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LIBRARIES "${PYTHON_LIBRARY}")
elseif(MSVC)
  # MSVC needs a little extra help finding the Python library
  find_package(PythonInterp)
  find_package(Python)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Run `nanobind.cmake_dir()` from Python to detect install location
execute_process(
  COMMAND
  "${PYTHON_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

# Now import nanobind from CMake
find_package(nanobind CONFIG REQUIRED)

# Eigen (required)
find_package(Eigen3 REQUIRED)

# Boost
find_package(Boost REQUIRED)

# Build the actual extension module
nanobind_add_module(
  pywrapper_robust_pca_impl
  NB_STATIC # Build static libnanobind (the extension module itself remains a shared library)
  src/pywrapper_robust_pca_impl.cpp
  src/impl/src/vanilla_pca.cpp
  src/impl/src/nb_convert.cpp
  src/impl/src/stat_utils.cpp
)

target_include_directories(
  pywrapper_robust_pca_impl
  PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  src/impl/include
)

target_link_libraries(
  pywrapper_robust_pca_impl
  PRIVATE
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  Eigen3::Eigen
)

# Install directive for scikit-build
install(TARGETS pywrapper_robust_pca_impl LIBRARY DESTINATION .)
