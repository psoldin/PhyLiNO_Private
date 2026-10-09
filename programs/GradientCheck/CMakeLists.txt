add_executable(GradientCheck GradientCheck.cpp)
target_link_libraries(GradientCheck LINK_PUBLIC ${ROOT_LIBRARIES} likelihood utilities io results doublechooz linearregression icecube)
set_target_properties(GradientCheck PROPERTIES LINKER_LANGUAGE CXX)
