cmake_minimum_required(VERSION 3.0.0)

project(Myprog)

find_package(OpenMP)
if (OpenMP_FOUND)
	message(STATUS "Link OpenMP")
	else() 
	message(STATUS "OpenMP not found!")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLUGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLUGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EX_ELINKER_FLAGS "${CMAKE_EX_ELINKER_FLAGS} ${OpenMP_EX_ELINKER_FLAGS}")

add_executable(${PROJECT_NAME} src/main.cpp)