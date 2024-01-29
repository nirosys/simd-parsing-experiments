cmake_minimum_required(VERSION 3.10)

project(SimdParsing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-march=native)

add_subdirectory(commanum)
