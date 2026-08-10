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

#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ana::ic {

  namespace {

    constexpr std::uint32_t kThreadsPerGroup = 256;  // must match every kernel

    struct MetalState {
      id<MTLDevice> dev;

      // Guards everything below. Scan workers build their Fits -- and therefore
      // their sessions -- concurrently, so the warmup paths that populate these
      // run on several threads at once. MTLDevice itself is thread-safe; these
      // C++ containers are not. The hot path (dispatch) takes no lock.
      std::mutex mutex;

      // deque, not vector: dispatch reads a session's alias rows while another
      // thread may still be appending columns.
      std::deque<id<MTLBuffer>>                        columns;   // shared column buffers
      std::unordered_map<const void*, int>             colCache;  // source ptr -> index into columns
      std::unordered_map<std::string,
                         id<MTLComputePipelineState>>  pipelines; // kernel name -> pso

      // Test-visible counters (see GpuBackend). Atomic rather than derived from
      // the containers above, so reading one needs no lock and cannot throw.
      std::atomic<std::size_t> columnCount{0};
      std::atomic<std::size_t> kernelCount{0};
      std::atomic<std::size_t> liveOutputs{0};
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
      std::deque<SessionRow> rows;

      // This sample's own command queue, the Metal analogue of a CUDA stream.
      // Samples are evaluated concurrently and scan workers run whole fits
      // concurrently, so their command buffers must be able to overlap rather
      // than queue up behind one another.
      id<MTLCommandQueue> queue;

      // The last command buffer committed on that queue, not yet waited on.
      // dispatch() no longer blocks: one likelihood evaluation is a chain of
      // kernels (flux, gather, ssq, gather) where only the last result is read
      // on the host, and a queue executes its command buffers in the order they
      // were committed. So the waits in between bought nothing but their own
      // latency, several per evaluation per sample. contents() is where the
      // host actually needs the data, and that is where the wait happens now.
      id<MTLCommandBuffer> pending = nil;
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

      auto* s = new MetalState;
      s->dev  = dev;
      m_State = s;
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

  std::size_t MetalBackend::column_count() const noexcept {
    auto* s = static_cast<MetalState*>(m_State);
    return s->columnCount.load(std::memory_order_relaxed);
  }

  std::size_t MetalBackend::kernel_compile_count() const noexcept {
    auto* s = static_cast<MetalState*>(m_State);
    return s->kernelCount.load(std::memory_order_relaxed);
  }

  std::size_t MetalBackend::live_output_count() const noexcept {
    auto* s = static_cast<MetalState*>(m_State);
    return s->liveOutputs.load(std::memory_order_relaxed);
  }

  MetalSession::MetalSession(std::shared_ptr<MetalBackend> backend)
    : m_Backend(std::move(backend)) {
    if (!m_Backend)
      throw std::runtime_error("MetalSession: null backend");

    auto* b = static_cast<MetalState*>(m_Backend->m_State);
    auto* s = new MetalSessionState;
    s->queue = [b->dev newCommandQueue];
    if (!s->queue) {
      delete s;
      // Deliberately fatal rather than falling back to the CPU path for this one
      // sample: scan points that silently used different code paths would not be
      // comparable with each other.
      throw std::runtime_error("MetalSession: could not create a command queue");
    }
    m_State = s;
  }

  MetalSession::~MetalSession() {
    auto* s = static_cast<MetalSessionState*>(m_State);
    if (!s) return;
    auto* b = static_cast<MetalState*>(m_Backend->m_State);
    // A committed command buffer may still be writing into this session's
    // buffers, which ARC is about to release.
    if (s->pending != nil) {
      [s->pending waitUntilCompleted];
      s->pending = nil;
    }
    for (const auto& row : s->rows)
      if (row.owning) b->liveOutputs.fetch_sub(1, std::memory_order_relaxed);
    // ARC releases every row; the backend's columns survive because it holds its
    // own strong reference to them.
    delete s;
  }

  void MetalSession::ensure_kernel(const char* name, const char* source) {
    auto* b = static_cast<MetalState*>(m_Backend->m_State);
    // Held across the compile: concurrent sessions asking for the same kernel
    // must not each build a pipeline for it.
    const std::scoped_lock lock(b->mutex);

    const std::string key(name);
    if (b->pipelines.count(key)) return;

    @autoreleasepool {
      NSError*            err  = nil;
      MTLCompileOptions*  opts = [MTLCompileOptions new];
      // Not the default (fast math is on unless you say otherwise). Two reasons,
      // both about the FP32 path's accuracy, which is what limits the fit's
      // resolution on Metal:
      //   1. fast math reassociates freely, which folds the Neumaier
      //      compensation term to a constant zero and deletes it -- the
      //      compensated sums below would be dead code.
      //   2. it selects the low-accuracy exp/exp2, evaluated once per event in
      //      every flux kernel. That error is per-event and no summation
      //      algorithm can recover it.
      opts.fastMathEnabled = NO;
      id<MTLLibrary> lib   = [b->dev newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                options:opts
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
      b->kernelCount.fetch_add(1, std::memory_order_relaxed);
    }
  }

  int MetalSession::upload_column(const double* data, std::size_t n) {
    auto* b = static_cast<MetalState*>(m_Backend->m_State);
    auto* s = static_cast<MetalSessionState*>(m_State);

    const std::scoped_lock lock(b->mutex);

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
      b->columnCount.fetch_add(1, std::memory_order_relaxed);
      return alias_row(s, buf);
    }
  }

  int MetalSession::upload_offsets(const std::size_t* data, std::size_t n) {
    auto* b = static_cast<MetalState*>(m_Backend->m_State);
    auto* s = static_cast<MetalSessionState*>(m_State);

    const std::scoped_lock lock(b->mutex);

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
      b->columnCount.fetch_add(1, std::memory_order_relaxed);
      return alias_row(s, buf);
    }
  }

  // `readback` is deliberately ignored: every buffer is MTLResourceStorageModeShared,
  // so contents() hands back the buffer itself and there is no copy to skip. The
  // flag only pays off on a discrete GPU (see CudaSession::alloc_output).
  int MetalSession::alloc_output(std::size_t n, bool /*readback*/) {
    auto* b = static_cast<MetalState*>(m_Backend->m_State);
    auto* s = static_cast<MetalSessionState*>(m_State);
    @autoreleasepool {
      id<MTLBuffer> buf = [b->dev newBufferWithLength:n * sizeof(float)
                                             options:MTLResourceStorageModeShared];
      std::memset(buf.contents, 0, n * sizeof(float));
      const int handle = static_cast<int>(s->rows.size());
      s->rows.push_back(SessionRow{buf, true});
      b->liveOutputs.fetch_add(1, std::memory_order_relaxed);
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

      id<MTLCommandBuffer>         cb  = [s->queue commandBuffer];
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
      s->pending = cb;
    }
  }

  const float* MetalSession::contents(int handle) const noexcept {
    auto* s = static_cast<MetalSessionState*>(m_State);
    // Everything committed on this queue runs in order, so waiting on the last
    // command buffer waits for all of them.
    if (s->pending != nil) {
      [s->pending waitUntilCompleted];
      s->pending = nil;
    }
    return static_cast<const float*>(s->rows[handle].buf.contents);
  }

}  // namespace ana::ic
