cmake_minimum_required (VERSION 3.1)
 
# Define project name
project (tp4)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_executable(tp4 main.cpp Matrix.cpp)

target_include_directories (tp4 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (tp4 ${OpenCL_LIBRARY})


INCLUDE(CPack)



