# The Double Chooz experiment module. Sources stay in their historical locations
# (io/DoubleChooz, likelihood/DoubleChooz, results/DoubleChooz) but build into one
# optional library target.
set(files
    ../../io/DoubleChooz/DCInputOptions.h
    ../../io/DoubleChooz/DCInputOptions.cpp
    ../../io/DoubleChooz/DCDetectorPaths.h
    ../../io/DoubleChooz/DCDetectorPaths.cpp
    ../../io/DoubleChooz/DataBase.h
    ../../io/DoubleChooz/DataBase.cpp
    ../../io/DoubleChooz/Constants.h
    ../../io/DoubleChooz/DCOptions.h
    ../../io/DoubleChooz/StartingParameter.h
    ../../io/DoubleChooz/StartingParameter.cpp
    ../../io/Parameter.h
    ../../io/ReactorData.h
    ../../io/ReactorData.cpp
    ../../results/DoubleChooz/DCWriteResults.h
    Definitions.h
    SpectrumBase.h
    Calculate_Spectrum.h
    RangeOscillator.h
    OscillationData.h
    Oscillator.h
    Oscillator.cpp
    ThreeFlavorOscillation.h
    ThreeFlavorOscillation.cpp
    ReactorSpectrum.h
    ReactorSpectrum.cpp
    EnergyCorrection.h
    EnergyCorrection.cpp
    ShapeCorrection.h
    ShapeCorrection.cpp
    AccidentalBackground.h
    AccidentalBackground.cpp
    LithiumBackground.h
    LithiumBackground.cpp
    FastNBackground.h
    FastNBackground.cpp
    DNCBackground.h
    DNCBackground.cpp
    DCLikelihood.h
    DCLikelihood.cpp
    DCExperimentModule.h
    DCExperimentModule.cpp
)

add_library(doublechooz SHARED ${files})

target_include_directories(doublechooz PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(doublechooz PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../results)
target_include_directories(doublechooz PUBLIC ${Boost_INCLUDE_DIR})
target_include_directories(doublechooz PUBLIC ${ROOT_INCLUDE_DIR})

target_link_libraries(doublechooz PUBLIC io likelihood utilities ${Boost_LIBRARIES}
                      ROOT::Core ROOT::Tree ROOT::TreePlayer)
target_link_libraries(doublechooz PRIVATE nlohmann_json::nlohmann_json)

# ThreeFlavorOscillation carries omp simd pragmas.
if(OpenMP_CXX_FOUND)
    target_link_libraries(doublechooz PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(doublechooz PROPERTIES LINKER_LANGUAGE CXX)
