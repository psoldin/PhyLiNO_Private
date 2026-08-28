# The model layer holds every bit of physics in the explorer and contains no Qt,
# so it is its own library: the timing benchmark and the unit tests link it
# without pulling a GUI toolkit in, and the cluster build (no Qt) still gets it.
add_library(explorermodel STATIC ExplorerModel.cpp)
target_link_libraries(explorermodel PUBLIC io likelihood icecube)
target_include_directories(explorermodel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# ICComponentBreakdown.h is header-only and lives with the results writers,
# which export no target for it -- ictests includes it the same way.
target_include_directories(explorermodel PUBLIC ${CMAKE_SOURCE_DIR}/libraries/results/IceCube)

# What one evaluation costs, which decides the GUI's threading design.
add_executable(PhyLiNOExplorerBench Bench.cpp)
target_link_libraries(PhyLiNOExplorerBench PRIVATE explorermodel)
