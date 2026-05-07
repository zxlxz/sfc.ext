#pragma once

#ifdef __CUDACC__
#define _dev __device__
#else
#define _dev
#endif

namespace sfc::cuda {

using i8 = signed char;
using i16 = short;
using i32 = int;

using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;

using usize = decltype(sizeof(0));

}  // namespace sfc::cuda
