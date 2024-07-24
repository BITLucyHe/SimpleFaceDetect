cmake_minimum_required(VERSION 3.0)

project(main)

add_executable(main main.cpp)

find_package(OpenCV REQUIRED)

link_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(main ${OpenCV_LIBS})