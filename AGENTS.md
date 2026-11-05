# AGENTS.md

SFC is a zero-STL, Rust-inspired foundational library for modern C++23 applications focused on safety, determinism, and performance – built from scratch so you control every byte.

## Layout

- `src/sfc/<mod>/`: one directory per module
- `cmake/`: shared CMake configs
- `build/`: CMake output, don't edit
- `<file>.h` (header), `<file>.cc` (source), `<file>.cxx` (unit tests)

Directories:

```
cmake/
src/
  sfc/
    cuda/
    math/
build/
```

## Build and Test

```shell
cmake -B build                # configure
cmake --build build           # build
```

```shell
# run all tests
./build/bin/sfc-test

# list tests (gtest format)
./build/bin/sfc-cuda-test --gtest_list_tests
./build/bin/sfc-math-test --gtest_list_tests

# run one test: sfc::<mod>::test.<name>
./build/bin/sfc-cuda-test --gtest_filter sfc::cuda::test.global_grow
./build/bin/sfc-math-test --gtest_filter sfc::math::test.global_grow
```

## Conventions

- Don't include C++ standard headers or use `std::` types and functions.

## UnitTest

```cpp
#include "sfc/test/test.h"

namespace sfc::math::ndslice::test {    // replace `math::ndslice` with your module
  SFC_TEST(ndview_len) {                // defines a test
    sfc::assert_eq(a, b);               // a == b
    sfc::assert_ne(a, b);               // a != b
  }
}
```
