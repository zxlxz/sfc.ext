#include "sfc/math/fft.h"

#if !defined(__clang__) && !defined(__GNUC__)
#include <math.h>
#include <malloc.h>
#include <string.h>
#define __builtin_cosf   ::cosf
#define __builtin_sinf   ::sinf
#define __builtin_sqrtf  ::sqrtf
#define __builtin_memcpy ::memcpy
#define __builtin_alloca ::_alloca
#endif

namespace sfc::math::detail {

static void fft_swap(auto& a, auto& b) {
  auto tmp = a;
  a = b;
  b = tmp;
}

static auto fft_rot(u32 n, f32 SIGN) -> c32 {
  // a = SIGN * 2 * PI / n
  // w = (cos(a), sin(a))
  switch (n / 2) {
    case 0:  return c32{+1, 0};                                  // a = 0
    case 1:  return c32{-1, 0};                                  // a = SIGN * PI
    case 2:  return c32{0, SIGN};                                // a = SIGN * PI / 2
    case 3:  return c32{0.50000000000f, SIGN * 0.86602540378f};  // a = SIGN * PI / 3
    case 4:  return c32{0.70710678118f, SIGN * 0.70710678118f};  // a = SIGN * PI / 4
    case 6:  return c32{0.86602540378f, SIGN * 0.50000000000f};  // a = SIGN * PI / 6
    case 7:  return c32{0.90096886790f, SIGN * 0.43388373912f};  // a = SIGN * PI / 7
    case 8:  return c32{0.92387953251f, SIGN * 0.38268343236f};  // a = SIGN * PI / 8
    default: break;
  }

  const auto a = SIGN * (2 * PI) / n;
  const auto c = __builtin_cosf(a);
  const auto s = __builtin_sinf(a);
  return c32{c, s};
}

static void fft_blk2(c32 H[], int SIGN) {
  (void)SIGN;

  const auto x0 = H[0];
  const auto x1 = H[1];

  // w = [1, -1]
  H[0] = x0 + x1;
  H[1] = x0 - x1;
}

static void fft_blk4(c32 H[], f32 SIGN) {
  const auto x0 = H[0];
  const auto x1 = H[1];
  const auto x2 = H[2];
  const auto x3 = H[3];

  // w(+) = [1, +i, -1, -i]
  // w(-) = [1, -i, -1, +i]
  const auto s1 = c32{-x1.imag, +SIGN * x1.real};
  const auto s3 = c32{-x3.imag, -SIGN * x3.real};
  H[0] = x0 + x1 + x2 + x3;
  H[1] = x0 + s1 - x2 + s3;
  H[2] = x0 - x1 + x2 - x3;
  H[3] = x0 - s1 - x2 - s3;
}

static void fft_blk3(c32 H[], f32 SIGN) {
  // w(+) = [1, (-0.5, +0.86602540378), (-0.5, -0.86602540378)]
  // w(-) = [1, (-0.5, -0.86602540378), (-0.5, +0.86602540378)]
  const auto w1 = c32{-0.5f, +SIGN * 0.86602540378f};
  const auto w2 = c32{-0.5f, -SIGN * 0.86602540378f};

  const auto x0 = H[0];
  const auto x1 = H[1];
  const auto x2 = H[2];

  H[0] = x0 + x1 + x2;
  H[1] = x0 + w1 * x1 + w2 * x2;
  H[2] = x0 + w2 * x1 + w1 * x2;
}

static void fft_blk5(c32 H[], f32 SIGN) {
  const auto w1 = c32{+0.30901699437f, +SIGN * 0.95105651629f};
  const auto w2 = c32{-0.80901699437f, +SIGN * 0.58778525229f};
  const auto w3 = c32{-0.80901699437f, -SIGN * 0.58778525229f};
  const auto w4 = c32{+0.30901699437f, -SIGN * 0.95105651629f};

  const auto x0 = H[0];
  const auto x1 = H[1];
  const auto x2 = H[2];
  const auto x3 = H[3];
  const auto x4 = H[4];

  H[0] = x0 + x1 + x2 + x3 + x4;
  H[1] = x0 + w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4;
  H[2] = x0 + w2 * x1 + w4 * x2 + w1 * x3 + w3 * x4;
  H[3] = x0 + w3 * x1 + w1 * x2 + w4 * x3 + w2 * x4;
  H[4] = x0 + w4 * x1 + w3 * x2 + w2 * x3 + w1 * x4;
}

static void fft_blk6(c32 H[], f32 SIGN) {
  c32 A[] = {H[0], H[2], H[4]};
  c32 B[] = {H[1], H[3], H[5]};
  detail::fft_blk3(A, SIGN);
  detail::fft_blk3(B, SIGN);

  const auto w1 = c32{+0.5f, SIGN * 0.86602540378f};
  const auto w2 = c32{-0.5f, SIGN * 0.86602540378f};

  H[0] = A[0] + B[0];
  H[1] = A[1] + w1 * B[1];
  H[2] = A[2] + w2 * B[2];

  H[3] = A[0] - B[0];
  H[4] = A[1] - w1 * B[1];
  H[5] = A[2] - w2 * B[2];
}

static void fft_blk7(c32 H[], f32 SIGN) {
  const auto w1 = c32{+0.62348980186f, +SIGN * 0.78183148091f};
  const auto w2 = c32{-0.22252043448f, +SIGN * 0.97492791218f};
  const auto w3 = c32{-0.90096886790f, +SIGN * 0.43388373912f};
  const auto w4 = c32{-0.90096886790f, -SIGN * 0.43388373912f};
  const auto w5 = c32{-0.22252043448f, -SIGN * 0.97492791218f};
  const auto w6 = c32{+0.62348980186f, -SIGN * 0.78183148091f};

  const auto x0 = H[0];
  const auto x1 = H[1];
  const auto x2 = H[2];
  const auto x3 = H[3];
  const auto x4 = H[4];
  const auto x5 = H[5];
  const auto x6 = H[6];

  H[0] = x0 + x1 + x2 + x3 + x4 + x5 + x6;
  H[1] = x0 + w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 + w6 * x6;
  H[2] = x0 + w2 * x1 + w4 * x2 + w6 * x3 + w1 * x4 + w3 * x5 + w5 * x6;
  H[3] = x0 + w3 * x1 + w6 * x2 + w2 * x3 + w5 * x4 + w1 * x5 + w4 * x6;
  H[4] = x0 + w4 * x1 + w1 * x2 + w5 * x3 + w2 * x4 + w6 * x5 + w3 * x6;
  H[5] = x0 + w5 * x1 + w3 * x2 + w1 * x3 + w6 * x4 + w4 * x5 + w2 * x6;
  H[6] = x0 + w6 * x1 + w5 * x2 + w4 * x3 + w3 * x4 + w2 * x5 + w1 * x6;
}

static void fft_blk8(c32 H[], f32 SIGN) {
  c32 A[] = {H[0], H[2], H[4], H[6]};
  c32 B[] = {H[1], H[3], H[5], H[7]};
  detail::fft_blk4(A, SIGN);
  detail::fft_blk4(B, SIGN);

  const auto r = 0.70710678118f;
  const auto w1 = c32{+r, SIGN * r};
  const auto w2 = c32{+0, SIGN * 1};
  const auto w3 = c32{-r, SIGN * r};

  H[0] = A[0] + B[0];
  H[1] = A[1] + w1 * B[1];
  H[2] = A[2] + w2 * B[2];
  H[3] = A[3] + w3 * B[3];

  H[4] = A[0] - B[0];
  H[5] = A[1] - w1 * B[1];
  H[6] = A[2] - w2 * B[2];
  H[7] = A[3] - w3 * B[3];
}

static void fft_blk(u32 N, c32 H[], f32 SIGN) {
  switch (N) {
    case 1:  return;
    case 2:  detail::fft_blk2(H, SIGN); return;
    case 3:  detail::fft_blk3(H, SIGN); return;
    case 4:  detail::fft_blk4(H, SIGN); return;
    case 5:  detail::fft_blk5(H, SIGN); return;
    case 6:  detail::fft_blk6(H, SIGN); return;
    case 7:  detail::fft_blk7(H, SIGN); return;
    case 8:  detail::fft_blk8(H, SIGN); return;
    default: return;
  }
}

static void fft_direct(u32 N, const c32 X[], c32 Y[], f32 SIGN) {
  for (u32 k = 0; k < N; ++k) {
    Y[k] = c32{0, 0};
    for (u32 n = 0; n < N; ++n) {
      const auto a = SIGN * (2 * PI) * k * n / N;
      const auto w = c32{__builtin_cosf(a), __builtin_sinf(a)};
      Y[k] = Y[k] + X[n] * w;
    }
  }
}

static void fft_inplace(u32 N, c32 H[], f32 SIGN) {
  if (N <= 8) {
    detail::fft_blk(N, H, SIGN);
    return;
  }

  if (N % 2 != 0) {
    auto Y = (c32*)__builtin_alloca(sizeof(c32) * N);
    detail::fft_direct(N, H, Y, SIGN);
    __builtin_memcpy(H, Y, sizeof(c32) * N);
    return;
  }

  // 1. reverse bits in-place
  auto j = 0U;
  for (auto i = 0U; i < N; ++i) {
    if (i < j) {
      detail::fft_swap(H[i], H[j]);
    }
    auto m = N / 2;
    while (j >= m && m > 0) {
      j -= m;
      m /= 2;
    }
    j += m;
  }

  // 2. loop over stages
  for (auto n = 2U; n <= N; n *= 2) {
    const auto m = n / 2;
    const auto r = detail::fft_rot(n, SIGN);
    for (auto i = 0U; i < N; i += n) {
      auto A = &H[i];
      auto B = &H[i + m];
      auto w = c32{1, 0};
      for (auto j = 0U; j < m; ++j) {
        const auto u = A[j];
        const auto v = w * B[j];
        A[j] = u + v;
        B[j] = u - v;
        w = w * r;
      }
    }
  }
}

static void fft_outplace(u32 N, const c32 X[], c32 Y[], f32 SIGN, u32 KX) {
  if (N <= 8) {
    c32 H[8];
    for (auto i = 0U; i < N; ++i) {
      H[i] = X[i * KX];
    }
    detail::fft_blk(N, H, SIGN);
    for (auto i = 0U; i < N; ++i) {
      Y[i] = H[i];
    }
    return;
  }

  if (N % 2 != 0) {
    detail::fft_direct(N, X, Y, SIGN);
    return;
  }

  const auto A = Y;
  const auto B = Y + N / 2;
  detail::fft_outplace(N / 2, X + 0 * KX, A, SIGN, 2 * KX);
  detail::fft_outplace(N / 2, X + 1 * KX, B, SIGN, 2 * KX);

  const auto r = detail::fft_rot(N, SIGN);
  auto w = c32{1, 0};
  for (u32 i = 0; i < N / 2; ++i) {
    const auto a = A[i];
    const auto b = B[i];
    A[i] = a + w * b;
    B[i] = a - w * b;
    w = w * r;
  }
}

static void fft_c2c(const c32 X[], c32 Y[], u32 N, int DIR) {
  if (N == 0) return;

  if (X == Y) {
    detail::fft_inplace(N, Y, DIR);
  } else {
    detail::fft_outplace(N, X, Y, DIR, 1);
  }
}

}  // namespace sfc::math::detail
