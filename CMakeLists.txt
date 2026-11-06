cmake_minimum_required(VERSION 3.27)
project(SCATI)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")

# the executable is always build/SCATI, also when an IDE builds in its own folder (cmake-build-*): the scripts look for it there
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

add_executable(SCATI
        src/SCATI.cpp
        src/Inference.cpp
        src/Node.cpp
        src/Scores.cpp
        src/input.cpp
        src/Tree.cpp
        src/Tree_moves.cpp
)
