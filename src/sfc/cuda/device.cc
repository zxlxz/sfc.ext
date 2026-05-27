
#include "sfc/cuda/mod.inl"
#include "sfc/cuda/device.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

void init() {
  static auto err = ::cuInit(0);
  cuda::check_ret(err, "cuInit");
}

auto dev_count() -> int {
  cuda::init();

  auto cnt = 0;
  CHECK_RET(cuDeviceGetCount, &cnt);
  return cnt;
}

auto device_get() -> int {
  cuda::init();

  auto dev = 0;
  CHECK_RET(cuCtxGetDevice, &dev);
  return dev;
}

void device_set(int dev) {
  cuda::init();

  static auto _tls_dev = -1;
  if (_tls_dev == dev) {
    return;
  }

  // get primary context
  auto context = CUcontext{nullptr};
  CHECK_RET(cuDevicePrimaryCtxRetain, &context, dev);

  // set context current
  CHECK_RET(cuCtxSetCurrent, context);
  _tls_dev = dev;
}

void device_sync() {
  cuda::init();
  CHECK_RET(cuCtxSynchronize);
}

auto device_attribute(int dev, CUdevice_attribute attr_id) -> int {
  cuda::init();

  auto attr_val = int{0};
  CHECK_RET(cuDeviceGetAttribute, &attr_val, attr_id, dev);
  return attr_val;
}

auto device_total_mem(int dev) -> usize {
  cuda::init();

  auto bytes = usize{0};
  CHECK_RET(cuDeviceTotalMem_v2, &bytes, dev);
  return bytes;
}

auto device_name(int dev) -> const char* {
  cuda::init();

  static thread_local char buf[64] = {};
  CHECK_RET(cuDeviceGetName, buf, sizeof(buf), dev);
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
