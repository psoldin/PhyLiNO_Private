# The .proto schema for the (optional) protobuf result format, generated into
# the build tree. Kept next to the JSON writer in results/IceCube. IMPORT_DIRS
# pins the proto's own directory as the -I root so the generated ic_result.pb.*
# land directly in PROTOC_OUT_DIR instead of under a "../../results/..." path.
protobuf_generate(
    LANGUAGE cpp
    OUT_VAR IC_RESULT_PROTO_GENERATED
    PROTOS ${CMAKE_CURRENT_SOURCE_DIR}/../../results/IceCube/proto/ic_result.proto
    IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../../results/IceCube/proto
    PROTOC_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
)

# The IceCube diffuse-flux experiment module (one composite likelihood over the
# config-enabled analysis samples). Sources are split across
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
    ../../results/IceCube/ICWriteResultsProto.h
    ../../results/IceCube/ICWriteResultsProto.cpp
    ${IC_RESULT_PROTO_GENERATED}
    PowerlawFlux.h
    PowerlawFlux.cpp
    AtmosphericFlux.h
    AtmosphericFlux.cpp
    DetectorSystematics.h
    DetectorSystematics.cpp
    TemplateFlux.h
    TemplateFlux.cpp
    SAYLikelihood.h
    SAYLikelihood.cpp
    PoissonLikelihood.h
    PoissonLikelihood.cpp
    SampleLikelihood.h
    SampleLikelihood.cpp
    ICLikelihood.h
    ICLikelihood.cpp
    ICModule.h
    ICModule.cpp
    GpuBackend.h
    MetalBackend.h
    CudaBackend.h
)

# GPU backends for the per-event flux histograms, both behind the GpuBackend
# facade. Component kernels live in the flux .cpp files as source strings (MSL
# and CUDA-C) and are compiled through the backend at runtime. Each backend is
# the real implementation where its toolkit exists and a no-op stub
# (available()==false) otherwise, so the library links everywhere.

# Metal: real Obj-C++/Metal on Apple, stub elsewhere.
if(APPLE)
    list(APPEND files MetalBackend.mm)
    set_source_files_properties(MetalBackend.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    list(APPEND files MetalBackend_stub.cpp)
endif()

# CUDA: real driver-API + NVRTC implementation when the CUDA toolkit is found,
# stub otherwise. Kernels are runtime-compiled by NVRTC, so no nvcc/device
# compilation is needed here -- CudaBackend.cpp is an ordinary C++ TU linking
# the CUDA driver + NVRTC libraries.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    list(APPEND files CudaBackend.cpp)
else()
    list(APPEND files CudaBackend_stub.cpp)
endif()

add_library(icecube SHARED ${files})

if(APPLE)
    target_link_libraries(icecube PRIVATE "-framework Metal" "-framework Foundation")
endif()

if(CUDAToolkit_FOUND)
    target_link_libraries(icecube PRIVATE CUDA::cuda_driver CUDA::nvrtc)
endif()

target_include_directories(icecube PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(icecube PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../results)
target_include_directories(icecube PUBLIC ${Boost_INCLUDE_DIR})
target_include_directories(icecube PRIVATE ${ROOT_INCLUDE_DIR})
target_include_directories(icecube PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(icecube PUBLIC io likelihood utilities ${Boost_LIBRARIES}
                      ROOT::Core ROOT::Tree ROOT::TreePlayer)
target_link_libraries(icecube PRIVATE nlohmann_json::nlohmann_json
                      Arrow::arrow_shared Parquet::parquet_shared
                      protobuf::libprotobuf)

# The per-bin flux and SAY ssq loops carry omp pragmas; without this target they
# are silently compiled away.
if(OpenMP_CXX_FOUND)
    target_link_libraries(icecube PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(icecube PROPERTIES LINKER_LANGUAGE CXX)
