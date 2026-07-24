// Shared Metal compute backend. The only Obj-C++/Metal translation unit in the
// icecube library; everything else talks to it through the pure-C++ facade in
// MetalBackend.h. Compiled only on Apple platforms (guarded in CMakeLists.txt),
// with ARC.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "MetalBackend.h"

#include "../../io/IceCube/ICConstants.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ana::ic {

  namespace {

    constexpr std::uint32_t kThreadsPerGroup = 256;  // must match every kernel

    struct MetalState {
      id<MTLDevice>       dev;
      id<MTLCommandQueue> queue;

      std::vector<id<MTLBuffer>>                       buffers;   // handle -> buffer
      std::unordered_map<const void*, int>             colCache;  // source ptr -> handle
      std::unordered_map<std::string,
                         id<MTLComputePipelineState>>  pipelines; // kernel name -> pso
    };

  }  // namespace

  MetalBackend::MetalBackend() {
    @autoreleasepool {
      id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
      if (!dev)
        throw std::runtime_error("MetalBackend: no Metal device available");

      auto* s  = new MetalState;
      s->dev   = dev;
      s->queue = [dev newCommandQueue];
      m_State  = s;
    }
  }

  MetalBackend::~MetalBackend() {
    delete static_cast<MetalState*>(m_State);
  }

  bool MetalBackend::available() noexcept {
    @autoreleasepool {
      return MTLCreateSystemDefaultDevice() != nil;
    }
  }

  void MetalBackend::ensure_kernel(const char* name, const char* source) {
    auto* s = static_cast<MetalState*>(m_State);
    const std::string key(name);
    if (s->pipelines.count(key)) return;

    @autoreleasepool {
      NSError*       err = nil;
      id<MTLLibrary> lib = [s->dev newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                options:[MTLCompileOptions new]
                                                  error:&err];
      if (!lib)
        throw std::runtime_error(std::string("MetalBackend: compile of '") + name + "' failed: " +
                                 (err ? err.localizedDescription.UTF8String : "unknown"));

      id<MTLFunction>             fn  = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
      if (!fn)
        throw std::runtime_error(std::string("MetalBackend: kernel '") + name + "' not found in source");

      id<MTLComputePipelineState> pso = [s->dev newComputePipelineStateWithFunction:fn error:&err];
      if (!pso)
        throw std::runtime_error(std::string("MetalBackend: pipeline for '") + name + "' failed: " +
                                 (err ? err.localizedDescription.UTF8String : "unknown"));
      s->pipelines[key] = pso;
    }
  }

  int MetalBackend::upload_column(const double* data, std::size_t n) {
    auto* s  = static_cast<MetalState*>(m_State);
    if (auto it = s->colCache.find(data); it != s->colCache.end())
      return it->second;

    @autoreleasepool {
      std::vector<float> f(n);
      for (std::size_t i = 0; i < n; ++i) f[i] = static_cast<float>(data[i]);
      id<MTLBuffer> buf = [s->dev newBufferWithBytes:f.data()
                                              length:n * sizeof(float)
                                             options:MTLResourceStorageModeShared];
      const int handle = static_cast<int>(s->buffers.size());
      s->buffers.push_back(buf);
      s->colCache[data] = handle;
      return handle;
    }
  }

  int MetalBackend::upload_offsets(const std::size_t* data, std::size_t n) {
    auto* s = static_cast<MetalState*>(m_State);
    if (auto it = s->colCache.find(data); it != s->colCache.end())
      return it->second;

    @autoreleasepool {
      std::vector<std::uint32_t> u(n);
      for (std::size_t i = 0; i < n; ++i) u[i] = static_cast<std::uint32_t>(data[i]);
      id<MTLBuffer> buf = [s->dev newBufferWithBytes:u.data()
                                              length:n * sizeof(std::uint32_t)
                                             options:MTLResourceStorageModeShared];
      const int handle = static_cast<int>(s->buffers.size());
      s->buffers.push_back(buf);
      s->colCache[data] = handle;
      return handle;
    }
  }

  int MetalBackend::alloc_output(std::size_t n) {
    auto* s = static_cast<MetalState*>(m_State);
    @autoreleasepool {
      id<MTLBuffer> buf = [s->dev newBufferWithLength:n * sizeof(float)
                                             options:MTLResourceStorageModeShared];
      std::memset(buf.contents, 0, n * sizeof(float));
      const int handle = static_cast<int>(s->buffers.size());
      s->buffers.push_back(buf);
      return handle;
    }
  }

  void MetalBackend::dispatch(const char* name,
                              const int*  inputs,
                              const int   n_inputs,
                              const void* params,
                              std::size_t params_len,
                              const int   hist,
                              const int   per_event) {
    auto* s = static_cast<MetalState*>(m_State);
    @autoreleasepool {
      id<MTLComputePipelineState> pso = s->pipelines.at(std::string(name));

      id<MTLCommandBuffer>         cb  = [s->queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      for (int i = 0; i < n_inputs; ++i)
        [enc setBuffer:s->buffers[inputs[i]] offset:0 atIndex:i];
      [enc setBytes:params length:params_len atIndex:n_inputs];
      [enc setBuffer:s->buffers[hist] offset:0 atIndex:n_inputs + 1];
      if (per_event >= 0)
        [enc setBuffer:s->buffers[per_event] offset:0 atIndex:n_inputs + 2];
      [enc dispatchThreadgroups:MTLSizeMake(io::ic::Constants::nBins, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kThreadsPerGroup, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    }
  }

  const float* MetalBackend::contents(int handle) const noexcept {
    auto* s = static_cast<MetalState*>(m_State);
    return static_cast<const float*>(s->buffers[handle].contents);
  }

}  // namespace ana::ic
