cmake_minimum_required(VERSION 3.25)
project(LearnOneDNN)

set(CMAKE_CXX_STANDARD 20)

find_package(dnnl 3.5.0 REQUIRED)

# Get a list of all .cpp files in the root directory
file(GLOB sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

# Create an executable target for each .cpp file
foreach (file_path ${sources})
    get_filename_component(filename "${file_path}" NAME_WE)
    add_executable("${filename}" "${file_path}")
    target_link_libraries("${filename}" dnnl)
endforeach ()