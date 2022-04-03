cmake_minimum_required (VERSION 3.1)
 
# Define project name
project (tp4)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/VectorAdd.cl
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

IF (WIN32) 
	set(OpenCL_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/CL)
	set(OpenCL_LIBRARY ${PROJECT_SOURCE_DIR}/lib/x86_64/opencl.lib)
ELSE()
	find_package(OpenCL REQUIRED)
ENDIF()

message(${OpenCL_INCLUDE_DIRS})
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_executable(tp4 main.cpp Matrix.cpp Matrix.hpp)

target_include_directories (tp4 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (tp4 ${OpenCL_LIBRARY})


INCLUDE(CPack)



