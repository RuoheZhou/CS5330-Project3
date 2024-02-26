cmake_minimum_required(VERSION 3.0.0)
project(project1 VERSION 0.1.0)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)


# Include directories for OpenCV and Boost
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# Add the executable and link against OpenCV and Boost libraries
add_executable(cleaned_frame task1_and_2.cpp include/filters.hpp filters.cpp)
target_link_libraries(cleaned_frame ${OpenCV_LIBS})
add_executable(colormap task3.cpp include/filters.hpp filters.cpp)
target_link_libraries(colormap ${OpenCV_LIBS})
add_executable(task4 task4.cpp include/filters.hpp filters.cpp)
target_link_libraries(task4 ${OpenCV_LIBS})
add_executable(task5 task5.cpp include/filters.hpp filters.cpp)
target_link_libraries(task5 ${OpenCV_LIBS})
add_executable(task6 task6.cpp include/filters.hpp filters.cpp)
target_link_libraries(task6 ${OpenCV_LIBS})
add_executable(task9 task9.cpp include/filters.hpp filters.cpp)
target_link_libraries(task9 ${OpenCV_LIBS})