cmake_minimum_required(VERSION 3.0.0)
project(project1 VERSION 0.1.0)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Boost and its components
find_package(Boost REQUIRED COMPONENTS filesystem)

# Include directories for OpenCV and Boost
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# Add the executable and link against OpenCV and Boost libraries
add_executable(task1 task1.cpp)
target_link_libraries(task1 ${OpenCV_LIBS} ${Boost_LIBRARIES})
add_executable(task3 task3.cpp)
target_link_libraries(task3 ${OpenCV_LIBS} ${Boost_LIBRARIES})
