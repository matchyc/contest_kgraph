
cmake_minimum_required (VERSION 3.12)
# set(CMAKE_CXX_COMPILER /usr/bin/g++-9)
project (KGRAPH)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Ofast -std=c++14 -fPIC -lmimalloc -mtune=native -ftree-vectorize -funroll-loops -ffast-math -mavx -mavx512f -mavx512vl")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=skylake-avx512 -Ofast -std=c++14 -fPIC -lmimalloc -mtune=skylake-avx512 -ftree-vectorize -funroll-loops -ffast-math -mavx -mavx512f -mavx512vl")
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -march=native -Ofast -std=c++14 -fPIC -lmimalloc -mtune=native -ftree-vectorize -funroll-loops -ffast-math -mavx -mavx512f")

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Boost REQUIRED COMPONENTS timer chrono system program_options)

find_package(mimalloc REQUIRED)

set(LIB_SRC kgraph.cpp metric.cpp)
include_directories(${KGRAPH_SOURCE_DIR})
# add_library(kgraph SHARED ${LIB_SRC})
# target_link_libraries(kgraph PUBLIC mimalloc)
add_library(kgrapha STATIC ${LIB_SRC})
# target_link_libraries(kgraph PUBLIC mimalloc)
set_target_properties(kgrapha PROPERTIES OUTPUT_NAME kgraph)
add_executable(index index.cpp)
# add_executable(search search.cpp)
# set(TOOLS index search)
set(TOOLS index)
foreach(TOOL ${TOOLS})
	target_link_libraries(${TOOL} kgrapha ${Boost_LIBRARIES} rt mimalloc)
endforeach(TOOL)
install (FILES kgraph.h kgraph-data.h DESTINATION include) 
install (TARGETS 
# kgraph 
kgrapha
	 LIBRARY DESTINATION lib
	 ARCHIVE DESTINATION lib)
