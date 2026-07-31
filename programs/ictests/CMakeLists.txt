add_executable(ICTests ICTests.cpp)
target_link_libraries(ICTests PRIVATE io icecube GTest::gtest_main)
# io's Boost include dir is PRIVATE (not propagated to consumers); ICTests
# needs it directly to include boost/property_tree headers for the
# parse_samples() test. Boost itself is already a link dependency of io.
target_include_directories(ICTests PRIVATE ${Boost_INCLUDE_DIR})

include(GoogleTest)
if(CMAKE_CROSSCOMPILING)
  gtest_add_tests(TARGET ICTests TEST_PREFIX "ICTests.")
else()
  gtest_discover_tests(ICTests TEST_PREFIX "ICTests.")
endif()
