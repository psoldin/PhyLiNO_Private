add_executable(ICTests ICTests.cpp)
target_link_libraries(ICTests PRIVATE io)
# io's Boost include dir is PRIVATE (not propagated to consumers); ICTests
# needs it directly to include boost/property_tree headers for the
# parse_samples() test. Boost itself is already a link dependency of io.
target_include_directories(ICTests PRIVATE ${Boost_INCLUDE_DIR})
add_test(NAME ICTests COMMAND ICTests)
