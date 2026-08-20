add_executable(KdeStats KdeStats.cpp)
target_link_libraries(KdeStats PRIVATE io icecube likelihood utilities)
set_target_properties(KdeStats PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(KdeStats PRIVATE ${Boost_INCLUDE_DIR})
