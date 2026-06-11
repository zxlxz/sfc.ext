#pragma once

#include <cuda_runtime_api.h>
#include "sfc/core.h"

namespace sfc::cuda {

using stream_t = cudaStream_t;

auto stream_new(unsigned int flags = 0) -> stream_t;
void stream_del(stream_t);
void stream_sync(stream_t);

void stream_set(stream_t);
auto stream_get() -> stream_t;

class Stream {
  stream_t _raw{nullptr};

 public:
  Stream() noexcept;
  ~Stream() noexcept;

  Stream(Stream&& other) noexcept;
  Stream& operator=(Stream&& other) noexcept;

  static auto create(u32 flags = 0) -> Stream;
  void sync();

 public:
  void run(auto& f) {
    cuda::stream_set(_raw);
    f();
    cuda::stream_set(nullptr);
  }
};

}  // namespace sfc::cuda
