#include "sfc/math/fft.h"

namespace sfc::math {

namespace imp {

static void fft_swap(auto& a, auto& b) {
  auto tmp = a;
  a = b;
  b = tmp;
}

static auto fft_rot(u32 n, f32 SIGN) -> c32 {
  switch (n / 2) {
    case 0:  return c32{1, 0};
    case 1:  return c32{1, 0};
    case 2:  return c32{0, SIGN};
    case 3:  return c32{0.50000000000f, SIGN * 0.86602540378f};
    case 4:  return c32{0.70710678118f, SIGN * 0.70710678118f};
    case 6:  return c32{0.86602540378f, SIGN * 0.50000000000f};
    case 7:  return c32{0.90096886790f, SIGN * 0.43388373912f};
    case 8:  return c32{0.92387953251f, SIGN * 0.38268343236f};
    default: break;
  }

  const auto a = SIGN * 2 * PI / n;
  const auto c = __builtin_cosf(a);
  const auto s = __builtin_sqrtf(1 - c * c);
  return c32{c, SIGN * s};
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
  imp::fft_blk3(A, SIGN);
  imp::fft_blk3(B, SIGN);

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
  imp::fft_blk4(A, SIGN);
  imp::fft_blk4(B, SIGN);

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

static auto fft_blk(u32 N, c32 H[], f32 SIGN) -> bool {
  switch (N) {
    case 1:  return true;
    case 2:  imp::fft_blk2(H, SIGN); return true;
    case 3:  imp::fft_blk3(H, SIGN); return true;
    case 4:  imp::fft_blk4(H, SIGN); return true;
    case 5:  imp::fft_blk5(H, SIGN); return true;
    case 6:  imp::fft_blk6(H, SIGN); return true;
    case 7:  imp::fft_blk7(H, SIGN); return true;
    case 8:  imp::fft_blk8(H, SIGN); return true;
    default: return false;
  }
}

}  // namespace imp

static auto fft_outplace(u32 N, const c32 X[], c32 Y[], f32 SIGN, u32 KX) -> bool {
  if (N <= 8) {
    c32 H[8];
    for (auto i = 0U; i < N; ++i) {
      H[i] = X[i * KX];
    }
    imp::fft_blk(N, H, SIGN);
    for (auto i = 0U; i < N; ++i) {
      Y[i] = H[i];
    }
    return true;
  }

  if (N % 2 != 0) {
    return false;
  }

  const auto A = Y;
  const auto B = Y + N / 2;
  math::fft_outplace(N / 2, X + 0 * KX, A, SIGN, 2 * KX);
  math::fft_outplace(N / 2, X + 1 * KX, B, SIGN, 2 * KX);

  const auto r = imp::fft_rot(N, SIGN);
  auto w = c32{1, 0};
  for (u32 i = 0; i < N / 2; ++i) {
    const auto a = A[i];
    const auto b = B[i];
    A[i] = a + w * b;
    B[i] = a - w * b;
    w = w * r;
  }
  return true;
}

static auto fft_inplace(u32 N, c32 H[], f32 SIGN) -> bool {
  if (N <= 8) {
    imp::fft_blk(N, H, SIGN);
    return true;
  }

  if (N % 2 != 0) {
    return false;
  }

  // 1. reverse bits in-place
  auto j = 0U;
  for (auto i = 0U; i < N; ++i) {
    if (i < j) {
      imp::fft_swap(H[i], H[j]);
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
    const auto r = imp::fft_rot(n, SIGN);
    for (auto i = 0U; i < N; i += n) {
      auto w = c32{1, 0};
      auto A = H + i;
      auto B = H + i + m;
      for (auto j = 0U; j < m; ++j) {
        const auto u = A[j];
        const auto v = w * B[j];
        A[j] = u + v;
        B[j] = u - v;
        w = w * r;
      }
    }
  }

  return true;
}

static auto fft_imp(const c32 X[], c32 Y[], u32 N, int DIR) -> bool {
  if (X == Y) {
    return math::fft_inplace(N, Y, DIR);
  } else {
    return math::fft_outplace(N, X, Y, DIR, 1);
  }
}

auto fft(NdSlice<c32, 1> x, NdSlice<c32, 1> y) -> bool {
  if (x._dims.x != y._dims.x) {
    return false;
  }

  const auto N = x._dims.x;
  const auto X = x._data;
  const auto Y = y._data;
  return math::fft_imp(X, Y, N, -1);
}

auto ifft(NdSlice<c32, 1> x, NdSlice<c32, 1> y) -> bool {
  if (x._dims.x != y._dims.x) {
    return false;
  }

  const auto N = x._dims.x;
  const auto X = x._data;
  const auto Y = y._data;
  return math::fft_imp(X, Y, N, +1);
}

auto fft_batch(NdSlice<c32, 2> x, NdSlice<c32, 2> y) -> bool {
  if (x._dims.x != y._dims.x) {
    return false;
  }
  if (x._dims.y != y._dims.y) {
    return false;
  }

  const auto M = x._dims.y;
  for (auto j = 0U; j < M; ++j) {
    auto vx = x[j];
    auto vy = y[j];
    if (!math::fft(vx, vy)) {
      return false;
    }
  }
  return true;
}

auto ifft_batch(NdSlice<c32, 2> x, NdSlice<c32, 2> y) -> bool {
  if (x._dims.x != y._dims.x) {
    return false;
  }
  if (x._dims.y != y._dims.y) {
    return false;
  }

  const auto M = x._dims.y;
  for (auto j = 0U; j < M; ++j) {
    auto vx = x[j];
    auto vy = y[j];
    if (!math::ifft(vx, vy)) {
      return false;
    }
  }
  return true;
}

}  // namespace sfc::math
