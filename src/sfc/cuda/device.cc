
#include <string.h>
#include "sfc/cuda/mod.inl"
#include "sfc/cuda/device.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

template <>
auto error_name(cudaError_t code) -> cstr_t {
  const auto name = ::cudaGetErrorName(code);
  return name;
}

void init() {
  static auto err = cudaFree(nullptr);
  cuda::check_ret(err, "cudaFree(nullptr)");
}

auto dev_count() -> int {
  cuda::init();

  auto cnt = 0;
  CHECK_RET(cudaGetDeviceCount, &cnt);
  return cnt;
}

auto device_get() -> int {
  cuda::init();

  auto dev = 0;
  CHECK_RET(cudaGetDevice, &dev);
  return dev;
}

void device_set(int dev) {
  cuda::init();

  static auto _tls_dev = -1;
  if (_tls_dev == dev) {
    return;
  }

  CHECK_RET(cudaSetDevice, dev);
  _tls_dev = dev;
}

void device_sync() {
  cuda::init();
  CHECK_RET(cudaDeviceSynchronize);
}

auto device_total_mem(int dev) -> usize {
  cuda::init();

  auto prop = cudaDeviceProp{};
  CHECK_RET(cudaGetDeviceProperties, &prop, dev);
  return prop.totalGlobalMem;
}

auto device_name(int dev) -> const char* {
  cuda::init();

  auto prop = cudaDeviceProp{};
  CHECK_RET(cudaGetDeviceProperties, &prop, dev);

  static thread_local char buf[256] = {};
  memcpy(buf, prop.name, sizeof(prop.name));
  return buf;
}

auto Device::current() -> Device {
  const auto dev = cuda::device_get();
  return Device{dev};
}

auto Device::name() const -> const char* {
  return cuda::device_name(id);
}

auto Device::total_memory() const -> usize {
  return cuda::device_total_mem(id);
}

}  // namespace sfc::cuda
