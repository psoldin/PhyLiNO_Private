# The IceCube tracks-only diffuse-flux experiment module. Sources are split across
# io/IceCube, likelihood/IceCube and results/IceCube but build into one optional
# library target. Needs Apache Arrow + Parquet for the MC baseline parquet loader.
set(files
    ../../io/IceCube/ICParameter.h
    ../../io/IceCube/ICConstants.h
    ../../io/IceCube/ICSample.h
    ../../io/IceCube/ICInputOptions.h
    ../../io/IceCube/ICInputOptions.cpp
    ../../io/IceCube/ICDataBase.h
    ../../io/IceCube/ICDataBase.cpp
    ../../results/IceCube/ICWriteResults.h
    PowerlawFlux.h
    PowerlawFlux.cpp
    AtmosphericFlux.h
    AtmosphericFlux.cpp
    DetectorSystematics.h
    DetectorSystematics.cpp
    MuonTemplate.h
    MuonTemplate.cpp
    SAYLikelihood.h
    SAYLikelihood.cpp
    ICLikelihood.h
    ICLikelihood.cpp
    ICModule.h
    ICModule.cpp
    ICMetalPowerlaw.h
)

# Metal GPU backend for the powerlaw flux: the real Obj-C++/Metal implementation
# on Apple, a no-op stub (available()==false) everywhere else.
if(APPLE)
    list(APPEND files ICMetalPowerlaw.mm)
    set_source_files_properties(ICMetalPowerlaw.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    list(APPEND files ICMetalPowerlaw_stub.cpp)
endif()

add_library(icecube SHARED ${files})

if(APPLE)
    target_link_libraries(icecube PRIVATE "-framework Metal" "-framework Foundation")
endif()

target_include_directories(icecube PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(icecube PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../results)
target_include_directories(icecube PUBLIC ${Boost_INCLUDE_DIR})
target_include_directories(icecube PRIVATE ${ROOT_INCLUDE_DIR})

target_link_libraries(icecube PUBLIC io likelihood utilities ${Boost_LIBRARIES}
                      ROOT::Core ROOT::Tree ROOT::TreePlayer)
target_link_libraries(icecube PRIVATE nlohmann_json::nlohmann_json
                      Arrow::arrow_shared Parquet::parquet_shared)

set_target_properties(icecube PROPERTIES LINKER_LANGUAGE CXX)
