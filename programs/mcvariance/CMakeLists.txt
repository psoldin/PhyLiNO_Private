add_executable(McVariance McVariance.cpp)
target_link_libraries(McVariance PRIVATE io icecube)
target_include_directories(McVariance PRIVATE
    ${CMAKE_SOURCE_DIR}/libraries/io/IceCube
    ${CMAKE_SOURCE_DIR}/libraries/likelihood/IceCube
    ${Boost_INCLUDE_DIR})
