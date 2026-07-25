add_executable(ICTests ICTests.cpp)
target_link_libraries(ICTests PRIVATE io)
add_test(NAME ICTests COMMAND ICTests)
