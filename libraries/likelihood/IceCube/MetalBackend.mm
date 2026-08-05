// Shared Metal compute backend. The only Obj-C++/Metal translation unit in the
// icecube library; everything else talks to it through the pure-C++ facade in
// MetalBackend.h. Compiled only on Apple platforms (guarded in CMakeLists.txt),
// with ARC.
//
// Split of state, mirroring CudaBackend.cpp: MetalState is process-wide and
// holds what is identical for every sample and every fit (device, command
// queue, compiled pipelines, uploaded MC columns); MetalSessionState holds one
// sample's output buffers and its handle table.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "MetalBackend.h"

#include "../../io/IceCube/ICConstants.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ana::ic {

  namespace {

    constexpr std::uint32_t kThreadsPerGroup = 256;  // must match every kernel

    struct MetalState {
      id<MTLDevice>       dev;
      id<MTLCommandQueue> queue;

      std::vector<id<MTLBuffer>>                       columns;   // shared column buffers
      std::unordered_map<const void*, int>             colCache;  // source ptr -> index into columns
      std::unordered_map<std::string,
                         id<MTLComputePipelineState>>  pipelines; // kernel name -> pso
    };

    // One row of a session's flat handle table. `owning` distinguishes an output
    // this session allocated (and must be the only writer of) from an alias to a
    // column buffer the backend owns and every session shares. ARC keeps both
    // alive; the flag is what stops a session from treating a shared column as
    // its own scratch.
    struct SessionRow {
      id<MTLBuffer> buf;
      bool          owning = false;
    };

    struct MetalSessionState {
      std::vector<SessionRow> rows;
    };

    // Register a shared backend buffer in this session's table as a non-owning
    // alias, so callers see one flat handle space (see GpuSession).
    int alias_row(MetalSessionState* s, id<MTLBuffer> buf) {
      const int handle = static_cast<int>(s->rows.size());
      s->rows.push_back(SessionRow{buf, false});
      return handle;
    }

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

  std::shared_ptr<GpuSession> MetalBackend::create_session() {
    return std::make_shared<MetalSession>(
        std::static_pointer_cast<MetalBackend>(shared_from_this()));
  }

  MetalSession::MetalSession(std::shared_ptr<MetalBackend> backend)
    : m_Backend(std::move(backend)) {
    if (!m_Backend)
      throw std::runtime_error("MetalSession: null backend");
    m_State = new MetalSessionState;
  }

  MetalSession::~MetalSession() {
    // ARC releases every row; the backend's columns survive because it holds its
    // own strong reference to them.
    delete static_cast<MetalSessionState*>(m_State);
  }

  void MetalSession::ensure_kernel(const char* name, const char* source) {
    auto* b = static_cast<MetalState*>(m_Backend->m_State);
    const std::string key(name);
    if (b->pipelines.count(key)) return;

    @autoreleasepool {
      NSError*       err = nil;
      id<MTLLibrary> lib = [b->dev newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                options:[MTLCompileOptions new]
                                                  error:&err];
      if (!lib)
        throw std::runtime_error(std::string("MetalBackend: compile of '") + name + "' failed: " +
                                 (err ? err.localizedDescription.UTF8String : "unknown"));

      id<MTLFunction>             fn  = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
      if (!fn)
        throw std::runtime_error(std::string("MetalBackend: kernel '") + name + "' not found in source");

      id<MTLComputePipelineState> pso = [b->dev newComputePipelineStateWithFunction:fn error:&err];
      if (!pso)
        throw std::runtime_error(std::string("MetalBackend: pipeline for '") + name + "' failed: " +
                                 (err ? err.localizedDescription.UTF8String : "unknown"));
      b->pipelines[key] = pso;
    }
  }

  int MetalSession::upload_column(const double* data, std::size_t n) {
    auto* b = static_cast<MetalState*>(m_Backend->m_State);
    auto* s = static_cast<MetalSessionState*>(m_State);

    if (auto it = b->colCache.find(data); it != b->colCache.end())
      return alias_row(s, b->columns[it->second]);

    @autoreleasepool {
      std::vector<float> f(n);
      for (std::size_t i = 0; i < n; ++i) f[i] = static_cast<float>(data[i]);
      id<MTLBuffer> buf = [b->dev newBufferWithBytes:f.data()
                                              length:n * sizeof(float)
                                             options:MTLResourceStorageModeShared];
      b->colCache[data] = static_cast<int>(b->columns.size());
      b->columns.push_back(buf);
      return alias_row(s, buf);
    }
  }

  int MetalSession::upload_offsets(const std::size_t* data, std::size_t n) {
    auto* b = static_cast<MetalState*>(m_Backend->m_State);
    auto* s = static_cast<MetalSessionState*>(m_State);

    if (auto it = b->colCache.find(data); it != b->colCache.end())
      return alias_row(s, b->columns[it->second]);

    @autoreleasepool {
      std::vector<std::uint32_t> u(n);
      for (std::size_t i = 0; i < n; ++i) u[i] = static_cast<std::uint32_t>(data[i]);
      id<MTLBuffer> buf = [b->dev newBufferWithBytes:u.data()
                                              length:n * sizeof(std::uint32_t)
                                             options:MTLResourceStorageModeShared];
      b->colCache[data] = static_cast<int>(b->columns.size());
      b->columns.push_back(buf);
      return alias_row(s, buf);
    }
  }

  int MetalSession::alloc_output(std::size_t n) {
    auto* b = static_cast<MetalState*>(m_Backend->m_State);
    auto* s = static_cast<MetalSessionState*>(m_State);
    @autoreleasepool {
      id<MTLBuffer> buf = [b->dev newBufferWithLength:n * sizeof(float)
                                             options:MTLResourceStorageModeShared];
      std::memset(buf.contents, 0, n * sizeof(float));
      const int handle = static_cast<int>(s->rows.size());
      s->rows.push_back(SessionRow{buf, true});
      return handle;
    }
  }

  void MetalSession::dispatch(const char* name,
                              const int*  inputs,
                              const int   n_inputs,
                              const void* params,
                              std::size_t params_len,
                              const int   hist,
                              const int   per_event,
                              const std::size_t n_groups) {
    auto* b = static_cast<MetalState*>(m_Backend->m_State);
    auto* s = static_cast<MetalSessionState*>(m_State);
    @autoreleasepool {
      id<MTLComputePipelineState> pso = b->pipelines.at(std::string(name));

      id<MTLCommandBuffer>         cb  = [b->queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      for (int i = 0; i < n_inputs; ++i)
        [enc setBuffer:s->rows[inputs[i]].buf offset:0 atIndex:i];
      [enc setBytes:params length:params_len atIndex:n_inputs];
      [enc setBuffer:s->rows[hist].buf offset:0 atIndex:n_inputs + 1];
      if (per_event >= 0)
        [enc setBuffer:s->rows[per_event].buf offset:0 atIndex:n_inputs + 2];
      [enc dispatchThreadgroups:MTLSizeMake(n_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kThreadsPerGroup, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    }
  }

  const float* MetalSession::contents(int handle) const noexcept {
    auto* s = static_cast<MetalSessionState*>(m_State);
    return static_cast<const float*>(s->rows[handle].buf.contents);
  }

}  // namespace ana::ic
