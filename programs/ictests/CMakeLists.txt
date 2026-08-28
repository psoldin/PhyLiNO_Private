add_executable(ICTests ICTests.cpp Df64Test.cpp InputOptionsTests.cpp ExplorerModelTests.cpp)
# Arrow/Parquet are PRIVATE to `icecube`; ICTests links them itself to write the
# small parquet fixtures the ICDataBase read tests load.
target_link_libraries(ICTests PRIVATE io icecube GTest::gtest_main
                      Arrow::arrow_shared Parquet::parquet_shared)
# io's Boost include dir is PRIVATE (not propagated to consumers); ICTests
# needs it directly to include boost/property_tree headers for the
# parse_samples() test. Boost itself is already a link dependency of io.
target_include_directories(ICTests PRIVATE ${Boost_INCLUDE_DIR})
# The shared per-component breakdown both result writers use is header-only and
# lives with the writers; ICTests pins its component list directly.
target_include_directories(ICTests PRIVATE ${CMAKE_SOURCE_DIR}/libraries/results/IceCube)
# The explorer's marginalization is header-only; its tests need the header, not
# the explorermodel library (which would drag a likelihood in for arithmetic
# that takes a span and a Binning).
target_include_directories(ICTests PRIVATE ${CMAKE_SOURCE_DIR}/programs/ParamExplorer)

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
