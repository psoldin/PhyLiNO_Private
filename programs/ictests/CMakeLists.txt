add_executable(ICTests ICTests.cpp)
target_link_libraries(ICTests PRIVATE io icecube GTest::gtest_main)
# io's Boost include dir is PRIVATE (not propagated to consumers); ICTests
# needs it directly to include boost/property_tree headers for the
# parse_samples() test. Boost itself is already a link dependency of io.
target_include_directories(ICTests PRIVATE ${Boost_INCLUDE_DIR})

# The scan refinement is header-only and pulls in nothing from the fit, so its
# tests are a separate, fast target.
add_executable(AdaptiveGridTests AdaptiveGridTests.cpp AdaptiveScan1DTests.cpp ScanSeedsTests.cpp)
target_link_libraries(AdaptiveGridTests PRIVATE GTest::gtest_main)
target_include_directories(AdaptiveGridTests PRIVATE ${CMAKE_SOURCE_DIR}/programs/LLHFit)

include(GoogleTest)
if(CMAKE_CROSSCOMPILING)
  gtest_add_tests(TARGET ICTests TEST_PREFIX "ICTests.")
else()
  gtest_discover_tests(ICTests TEST_PREFIX "ICTests.")
endif()
