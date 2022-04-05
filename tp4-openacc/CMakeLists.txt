cmake_minimum_required (VERSION 3.1)

# Define project name
project (tp4-openacc)

include(FindOpenACC)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GTK_CFLAGS} ${OpenACC_CXX_FLAGS} -fcf-protection=none")

add_executable(tp4 main.cpp Matrix.cpp Matrix.hpp)

target_include_directories (tp4 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (tp4 ${OpenCL_LIBRARY})
