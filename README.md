# fmt-sycl

- `std::print`-like formatting for SYCL device kernels.
- Required C++20

- Currently only tested with DPC++

## Quick example

```cpp
#include "sycl_khr_print.hpp"
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  q.parallel_for(4, [=](sycl::id<1> i) {
    KHR_PRINTLN("work-item {} says {} pi={}",
        static_cast<int>(i), "hello", 3.14f);
  }).wait();
}
```

One possible ordering of the output:
```bash
work-item 0 says hello pi=3.14
work-item 2 says hello pi=3.14
work-item 1 says hello pi=3.14
work-item 3 says hello pi=3.14
```

## API

```cpp
sycl::khr::print<"format string">(args...);    // no trailing newline
sycl::khr::println<"format string">(args...);  // appends \n

// Convenience macros (avoid angle-bracket syntax)
KHR_PRINT("format string", args...);
KHR_PRINTLN("format string", args...);
```

### Difference from `std::print` / `std::println`

- We require a compile-time format string.
- Otherwise, we support the majority of `std::print` features.

- To ensure atomicity of the print:
  - We disable `dragonbox` by default (use `FMT_SYCL_RELAX_ATOMICITY` to opt in). This means some floats may be formatted differently by `std::print` compared to `sycl::khr::print`.
  - Some features are not implementable while keeping atomicity. If you use a non-atomic feature without the flag, you get a compile-time error with a workaround:

```
error: static assertion failed:
  This format string uses non-atomic features ({:b}, {:a}, {:^}, custom fill,
  {:#x} with signed int, etc.).
  Define FMT_SYCL_RELAX_ATOMICITY to enable
  (output may interleave across work-items).
```
## Build

```bash
# Single file, just include the header
icpx -fsycl -std=c++20 my_kernel.cpp -o my_kernel

# Enable non-atomic features (full std::format compatibility)
icpx -fsycl -std=c++20 -DFMT_SYCL_RELAX_ATOMICITY my_kernel.cpp -o my_kernel
```

## Tests

```bash
make test              # Run all tests
```
Or individually:
```bash
make test-interleave   # Atomicity test (multi-work-item)
make test-examples     # Format correctness (diff against std::format)
make test-fuzz         # Fuzz with random values
```
