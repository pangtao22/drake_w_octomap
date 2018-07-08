cmake_minimum_required(VERSION 3.9)
project(octomap_w_drake)

set(CMAKE_CXX_STANDARD 11)
# set(OCTOMAP_INCLUDE_DIRS /usr/local/include/octomap)

find_package(octomap REQUIRED)
find_package(drake CONFIG REQUIRED)
include_directories(${OCTOMAP_INCLUDE_DIRS})

add_executable(run_octomap_w_drake main.cpp)
target_link_libraries(run_octomap_w_drake
        ${OCTOMAP_LIBRARIES}
        drake::drake)