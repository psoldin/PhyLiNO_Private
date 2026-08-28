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

# The GUI is optional. Qt is a desktop-only dependency and the cluster has none,
# so its absence skips this target and leaves LLHFit and the model layer above
# untouched. Homebrew's Qt5 is keg-only, so a local build needs
#   cmake -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/qt@5 ..
find_package(Qt5 COMPONENTS Widgets Charts QUIET)
if(Qt5_FOUND)
  set(CMAKE_AUTOMOC ON)
  add_executable(PhyLiNOExplorer main.cpp MainWindow.cpp)
  target_link_libraries(PhyLiNOExplorer PRIVATE explorermodel Qt5::Widgets Qt5::Charts)
  message(STATUS "Qt5 ${Qt5_VERSION} found -- building PhyLiNOExplorer")
else()
  message(STATUS "Qt5 (Widgets, Charts) not found -- skipping PhyLiNOExplorer")
endif()
