cmake_minimum_required(VERSION 3.0.0)
project(project1 VERSION 0.1.0)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(include)

# Include directories for OpenCV and Boost
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Add the executable and link against OpenCV and Boost libraries
add_executable(task1 task1.cpp)
target_link_libraries(task1 ${OpenCV_LIBS})
add_executable(task3 task3.cpp include/filters.hpp)
target_link_libraries(task3 ${OpenCV_LIBS})
