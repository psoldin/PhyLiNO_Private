set(files
    LinRegInputOptions.h
    LinRegLikelihood.h
    LinRegLikelihood.cpp
    LinRegModule.h
    LinRegModule.cpp
)

add_library(linearregression SHARED ${files})

target_include_directories(linearregression PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(linearregression PRIVATE ${Boost_INCLUDE_DIR})
target_include_directories(linearregression PRIVATE ${ROOT_INCLUDE_DIR})

target_link_libraries(linearregression PUBLIC io likelihood utilities ${Boost_LIBRARIES})
target_link_libraries(linearregression PRIVATE nlohmann_json::nlohmann_json)

set_target_properties(linearregression PROPERTIES LINKER_LANGUAGE CXX)
