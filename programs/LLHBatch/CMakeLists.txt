set(files
       LLHBatch.C
)

add_executable(LLHBatch ${files})

target_link_libraries(LLHBatch LINK_PUBLIC ${ROOT_LIBRARIES} likelihood utilities io results icecube)
set_target_properties(LLHBatch PROPERTIES LINKER_LANGUAGE CXX)
