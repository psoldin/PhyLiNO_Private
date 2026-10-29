set(files
       LLHFit.C
       ScanCommon.cpp
       Scan1D.cpp
       Scan2D.cpp
       ScanPairs.cpp
)

add_executable(LLHFit ${files})

target_link_libraries(LLHFit LINK_PUBLIC ${ROOT_LIBRARIES} likelihood utilities io results doublechooz linearregression icecube ROOT::RooFit ROOT::RooStats ROOT::RooFitCore)
set_target_properties(LLHFit PROPERTIES LINKER_LANGUAGE CXX)
