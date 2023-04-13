cmake_minimum_required(VERSION 3.10)
project(cerceptron)

add_executable(circles_and_squares circles_and_squares.c)
target_compile_options(circles_and_squares PRIVATE -Wall -Wextra -Werror -pedantic-errors)

add_library(perceptron SHARED perceptron.c)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(circles_and_squares PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(circles_and_squares perceptron)
