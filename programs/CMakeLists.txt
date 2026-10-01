add_subdirectory(LLHFit)
add_subdirectory(ParamExplorer)
add_subdirectory(mcvariance)

if(BUILD_TESTING)
  add_subdirectory(ictests)
endif()
