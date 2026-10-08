cmake_minimum_required(VERSION 3.16)
project(neon_vio LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ORB_SLAM3_ROOT "" CACHE PATH "Pinned ORB-SLAM3 checkout with export_neon.patch applied")
if(NOT EXISTS "${ORB_SLAM3_ROOT}/include/System.h")
  message(FATAL_ERROR "Set ORB_SLAM3_ROOT to the patched ORB-SLAM3 checkout")
endif()
find_package(OpenCV 4.4 REQUIRED COMPONENTS core imgproc imgcodecs features2d calib3d highgui)
find_package(Eigen3 3.1 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(OpenSSL REQUIRED)
find_path(OPENSSL_MULTIARCH_INCLUDE_DIR openssl/opensslconf.h
  HINTS "${OPENSSL_INCLUDE_DIR}/${CMAKE_LIBRARY_ARCHITECTURE}")
find_package(Threads REQUIRED)
# build_compat.patch removes g2o's private -march=native: mixed AVX/SSE
# builds disagree on both Eigen object alignment and allocation/free policy.
# Also keep a shared explicit alignment ABI on all adapter targets.
add_compile_definitions(EIGEN_MAX_ALIGN_BYTES=16 EIGEN_MAX_STATIC_ALIGN_BYTES=16)
add_subdirectory("${ORB_SLAM3_ROOT}/Thirdparty/DBoW2" "dbow2")
add_subdirectory("${ORB_SLAM3_ROOT}/Thirdparty/g2o" "g2o")
# Upstream DBoW2 assumes Boost headers are in /usr/include. Propagate the
# imported target so a local, administrator-free prefix also works.
target_link_libraries(DBoW2 Boost::serialization)
file(GLOB ORB_SOURCES CONFIGURE_DEPENDS "${ORB_SLAM3_ROOT}/src/*.cc" "${ORB_SLAM3_ROOT}/src/*.cpp" "${ORB_SLAM3_ROOT}/src/CameraModels/*.cpp")
add_library(neon_orb_slam3 SHARED ${ORB_SOURCES})
target_compile_definitions(neon_orb_slam3 PUBLIC COMPILEDWITHC11)
target_include_directories(neon_orb_slam3 PUBLIC
  "${ORB_SLAM3_ROOT}" "${ORB_SLAM3_ROOT}/include"
  "${ORB_SLAM3_ROOT}/include/CameraModels" "${ORB_SLAM3_ROOT}/Thirdparty/Sophus"
  ${OpenCV_INCLUDE_DIRS})
if(OPENSSL_MULTIARCH_INCLUDE_DIR)
  target_include_directories(neon_orb_slam3 SYSTEM PRIVATE ${OPENSSL_MULTIARCH_INCLUDE_DIR})
endif()
target_link_libraries(neon_orb_slam3 PUBLIC DBoW2 g2o Eigen3::Eigen
  ${OpenCV_LIBS} pango_display Boost::serialization OpenSSL::Crypto Threads::Threads)
add_executable(mono_inertial_neon mono_inertial_neon.cc)
target_link_libraries(mono_inertial_neon PRIVATE neon_orb_slam3)
