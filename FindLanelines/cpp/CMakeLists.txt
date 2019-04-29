cmake_minimum_required(VERSION 3.1)

PROJECT(Lanelines)

set(CMAKE_BUILD_TYPE Debug)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS})

MACRO(add_example name)
  ADD_EXECUTABLE(${name} lanelines.cpp ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} )
ENDMACRO()

#add_example(lanelines_video)
add_example(video)
add_example(images)
